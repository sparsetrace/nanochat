"""
modal_train.py — train your nanochat fork on Modal (1x H100), launched from GitHub Actions.

Lives in the repo root of your nanochat fork.

How it works
------------
- GitHub Actions checks out your fork, then runs `modal run --detach ...`.
- The full repo is baked into the image and dependencies are installed the
  same way upstream does it: `uv sync --extra gpu` run inside the repo, with
  a Rust toolchain present (nanochat's tokenizer is a Rust extension built
  via maturin — installing deps outside the repo produces a broken env).
- Every commit triggers an image rebuild (~few minutes for uv sync). Correct
  beats fast; optimize later.
- All nanochat output (~/.cache/nanochat: data shards, tokenizer, checkpoints)
  lives on a persistent Volume. A background thread commits it every 15 min
  so a crash or timeout loses at most 15 min of checkpoints.
- Training output is streamed line-by-line into Modal logs (stderr merged),
  so the real traceback is always visible if the run dies.
- Final checkpoints are pushed to HuggingFace using the HF_TOKEN that GitHub
  Actions injects at launch (Secret.from_dict reads the runner's env).

Usage
-----
#   HF_TOKEN=hf_... modal run --detach modal_train.py --depth 12 --hf-repo yourname/nanochat
"""

import os
import subprocess
import threading
import time
from pathlib import Path

import modal

app = modal.App("nanochat")

GPU_CONFIG = "H100"       # Hopper: FA3 supported. For multi-GPU use e.g. "H100:8"
NPROC = 1                 # keep in sync with GPU_CONFIG; >1 switches to torchrun

CACHE_DIR = "/root/.cache/nanochat"   # nanochat writes everything here by default
REPO_DIR = "/root/nanochat"
VENV = f"{REPO_DIR}/.venv"

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
# Image: replicate upstream's reference setup exactly.
#   - Rust toolchain (rustbpe tokenizer builds via maturin during uv sync)
#   - full repo baked in (copy=True) so uv sync sees the project source
#   - `uv sync --extra gpu` inside the repo, same as the nanochat README
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "build-essential")
    .pip_install("huggingface_hub")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
    .env(
        {
            "PATH": "/root/.cargo/bin:/root/.local/bin:/usr/local/bin:/usr/bin:/bin",
            "OMP_NUM_THREADS": "1",
        }
    )
    .add_local_dir(
        ".",
        remote_path=REPO_DIR,
        copy=True,  # must be baked so uv sync can build the project at image time
        ignore=[".git", ".venv", "__pycache__", "*.pyc", ".github"],
    )
    .run_commands(f"cd {REPO_DIR} && uv sync --extra gpu")
)


def _periodic_volume_commit(stop: threading.Event, every_s: int = 900):
    """Flush Volume writes every 15 min so checkpoints survive a crash."""
    while not stop.wait(every_s):
        try:
            ckpt_vol.commit()
            print("[modal_train] volume checkpoint committed")
        except Exception as e:  # never kill training over a commit hiccup
            print(f"[modal_train] volume commit failed (non-fatal): {e}")


def _run_streamed(cmd: str):
    """Run cmd, streaming merged stdout+stderr line-by-line into Modal logs.

    Guarantees the child's traceback is visible if it dies (unlike bare
    subprocess.run, whose stderr can get lost in log routing).
    """
    print(f"[modal_train] running: {cmd}", flush=True)
    proc = subprocess.Popen(
        cmd,
        shell=True,
        executable="/bin/bash",
        cwd=REPO_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}: {cmd}")


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    cpu=8,
    memory=32768,           # 32 GiB; dataloader workers are the main consumer
    timeout=24 * 60 * 60,   # Modal's max
    retries=0,              # a retry restarts training from scratch = $$$
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
    os.chdir(REPO_DIR)
    run_name = f"d{depth}"

    # Background committer so mid-run checkpoints are durable
    stop = threading.Event()
    threading.Thread(target=_periodic_volume_commit, args=(stop,), daemon=True).start()

    python = f"{VENV}/bin/python"
    torchrun = f"{VENV}/bin/torchrun"

    # Sanity check: confirm training will import the fork's code (with your
    # modifications), through the same interpreter training uses.
    _run_streamed(
        f'{python} -c "import nanochat.gpt; '
        f"print('[sanity] gpt.py:', nanochat.gpt.__file__)\""
    )

    # NOTE: the trailing '--' separator is torchrun syntax only; plain python
    # argparse chokes on it (exit 2).
    if NPROC > 1:
        launcher = f"{torchrun} --standalone --nproc_per_node={NPROC} -m scripts.base_train --"
    else:
        launcher = f"{python} -m scripts.base_train"

    # NOTE: no --run flag on purpose — nanochat's default ('dummy') disables
    # wandb. To enable wandb: add WANDB_API_KEY to the secret and pass
    # --run=<name> via extra_args.
    cmd = (
        f"{launcher} "
        f"--depth={depth} "
        f"--model-tag={run_name} "
        f"--save-every={save_every} "
        f"{extra_args}"
    )
    t0 = time.time()
    try:
        _run_streamed(cmd)
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
                ckpt_dir = Path(CACHE_DIR) / "checkpoints"  # fallback: everything
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
