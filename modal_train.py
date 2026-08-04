"""
modal_train.py — train your nanochat fork on Modal (1x B200), launched from GitHub Actions.

Lives in the repo root of your nanochat fork.

How it works
------------
- GitHub Actions checks out your fork, then runs `modal run --detach ...`.
- Your working tree (the exact commit that triggered the workflow) is synced
  into the container via add_local_dir — no git clone, no image rebuild per commit.
- Dependencies are baked into the image from pyproject.toml/uv.lock, so they
  only rebuild when the lockfile changes.
- All nanochat output (~/.cache/nanochat: data shards, tokenizer, checkpoints)
  lives on a persistent Volume, so dataset downloads and checkpoints survive
  across runs. A background thread commits the Volume every 15 minutes so a
  mid-run crash doesn't lose checkpoints.
- Final checkpoints are pushed to HuggingFace using the HF_TOKEN that GitHub
  Actions injects at launch time (Secret.from_dict reads the runner's env).

Usage
-----
# From CI (see .github/workflows/train.yml), or locally:
#   HF_TOKEN=hf_... modal run --detach modal_train.py --depth 12 --hf-repo yourname/nanochat
"""

import os
import subprocess
import threading
import time
from pathlib import Path

import modal

app = modal.App("nanochat")

GPU_CONFIG = "B200"       # 192 GB HBM3e, ~$6.25/hr. For multi-GPU use e.g. "B200:4"
NPROC = 1                 # keep in sync with GPU_CONFIG; >1 switches to torchrun

CACHE_DIR = "/root/.cache/nanochat"   # nanochat writes everything here by default
REPO_DIR = "/root/nanochat"
VENV = "/root/deps/.venv"

# ---------------------------------------------------------------------------
# Persistent storage: dataset shards + tokenizer + checkpoints
# ---------------------------------------------------------------------------
ckpt_vol = modal.Volume.from_name("nanochat-cache", create_if_missing=True, version=2)

# ---------------------------------------------------------------------------
# HF token: read from the environment of whoever launches the run
# (GitHub Actions passes secrets.HF_TOKEN as env — no Modal secret needed).
# ---------------------------------------------------------------------------
hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})

# ---------------------------------------------------------------------------
# Image
#
# Two-stage trick for fast launches:
#   1. Deps are installed from pyproject.toml + uv.lock, baked into the image.
#      This layer only rebuilds when those two files change.
#   2. The full repo is attached with add_local_dir (no copy=True), which is
#      synced at container start — code changes never trigger an image rebuild.
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "build-essential")
    .pip_install("uv", "huggingface_hub")
    .add_local_file("pyproject.toml", "/root/deps/pyproject.toml", copy=True)
    .add_local_file("uv.lock", "/root/deps/uv.lock", copy=True)
    .run_commands("cd /root/deps && uv sync --extra gpu")
    .env({"OMP_NUM_THREADS": "1"})
    .add_local_dir(
        ".",
        remote_path=REPO_DIR,
        ignore=[".git", ".venv", "__pycache__", "*.pyc", ".github"],
    )
)


def _periodic_volume_commit(stop: threading.Event, every_s: int = 900):
    """Flush Volume writes every 15 min so checkpoints survive a crash."""
    while not stop.wait(every_s):
        try:
            ckpt_vol.commit()
            print("[modal_train] volume checkpoint committed")
        except Exception as e:  # never kill training over a commit hiccup
            print(f"[modal_train] volume commit failed (non-fatal): {e}")


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    cpu=8,
    memory=32768,           # 32 GiB; dataloader workers are the main consumer
    ephemeral_disk=80_000,  # 80 GiB scratch (Volume holds the durable data)
    timeout=36 * 60 * 60,   # 36 h ceiling
    retries=0,              # a retry restarts training from scratch = $$$. Investigate failures manually.
    volumes={CACHE_DIR: ckpt_vol},
    secrets=[hf_secret],
)
def train_nanochat(
    depth: int = 12,
    hf_repo: str = "",
    save_every: int = 1000,
    extra_args: str = "",
) -> dict:
    """
    Pretrain nanochat at a given depth, checkpoint to the Volume, push to HF.

    depth      : transformer depth (the one nanochat complexity dial)
    hf_repo    : HF repo id e.g. "yourname/nanochat"; empty string skips upload
    save_every : checkpoint interval in steps (persisted to the Volume)
    extra_args : extra CLI flags, e.g. "--device-batch-size=16"
    """

    # sanity check: confirm we're importing the fork's code, not a stale copy
    os.chdir(REPO_DIR)
    run_name = f"d{depth}"

    # Background committer so mid-run checkpoints are durable
    stop = threading.Event()
    committer = threading.Thread(
        target=_periodic_volume_commit, args=(stop,), daemon=True
    )
    committer.start()

    python = f"{VENV}/bin/python"
    torchrun = f"{VENV}/bin/torchrun"
    if NPROC > 1:
        launcher = f"{torchrun} --standalone --nproc_per_node={NPROC} -m scripts.base_train --"
    else:
        launcher = f"{python} -m scripts.base_train --"


    sanity = (
        f'{python} -c "import nanochat.gpt; '
        f"print('[sanity] gpt.py:', nanochat.gpt.__file__)\" "
    )
    cmd = (
        f"{sanity} && "
        f"{launcher} "
        f"--depth={depth} "
        f"--model-tag={run_name} "
        f"--run={run_name} "
        f"--save-every={save_every} "
        f"{extra_args}"
    )
    print(f"[modal_train] running: {cmd}")
    t0 = time.time()
    try:
        subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")
    finally:
        stop.set()
        ckpt_vol.commit()  # always flush whatever we have, even on failure
    print(f"[modal_train] training finished in {(time.time() - t0) / 3600:.2f} h")

    # ── Push checkpoints to HuggingFace ──────────────────────────────────────
    if hf_repo:
        from huggingface_hub import HfApi

        token = os.environ.get("HF_TOKEN", "")
        if not token:
            print("[modal_train] WARNING: HF_TOKEN empty, skipping upload")
        else:
            # nanochat saves under ~/.cache/nanochat/checkpoints/<model-tag>/
            ckpt_dir = Path(CACHE_DIR) / "checkpoints" / run_name
            if not ckpt_dir.exists():
                # fall back: grab whatever checkpoints exist
                ckpt_dir = Path(CACHE_DIR) / "checkpoints"
            api = HfApi(token=token)
            api.create_repo(repo_id=hf_repo, private=True, exist_ok=True)
            api.upload_folder(
                repo_id=hf_repo,
                folder_path=str(ckpt_dir),
                path_in_repo=f"checkpoints/{run_name}",
            )
            print(f"[modal_train] pushed {ckpt_dir} -> {hf_repo}/checkpoints/{run_name}")

    return {"ok": True, "run": run_name, "hf_repo": hf_repo}


@app.local_entrypoint()
def main(
    depth: int = 12,
    hf_repo: str = "",
    save_every: int = 1000,
    extra_args: str = "",
):
    train_nanochat.remote(
        depth=depth, hf_repo=hf_repo, save_every=save_every, extra_args=extra_args
    )
