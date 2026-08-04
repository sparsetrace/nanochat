"""
modal_train.py — train your nanochat fork on Modal (1x H100), launched from GitHub Actions.

Lives in the repo root of your nanochat fork.

How it works
------------
- GitHub Actions checks out your fork, runs `modal deploy`, then spawns the
  training function and exits. The run is fully detached from the runner —
  no timeout or cancellation on the GitHub side can touch it.
- The full repo is baked into the image and dependencies are installed the
  same way upstream does it: `uv sync --extra gpu` run inside the repo, with
  a Rust toolchain present (nanochat's tokenizer is a Rust extension built
  via maturin). Every commit rebuilds the image (~few minutes of uv sync).
- All nanochat output (~/.cache/nanochat: data shards, tokenizer, checkpoints,
  training logs) lives on a persistent Volume. A background thread commits it
  every 15 min so a crash loses at most 15 min of progress.
- Training output is streamed line-by-line into Modal logs AND teed to a log
  file; after training the step lines are parsed into a CSV (stdlib only).
  Checkpoints + log + CSV are pushed to HuggingFace at the end.
- HF auth uses the HF_TOKEN that GitHub Actions injects at launch
  (Secret.from_dict reads the runner's env at deploy time).
- Default window_pattern is "L" (full causal context, no sliding windows):
  we are running architecture experiments and want the most expressive,
  mask-minimal configuration; SSSL is a wall-clock optimization we don't
  need. Pass window_pattern="SSSL" to restore upstream behavior.

Usage
-----
# CI does: modal deploy modal_train.py && spawn train_nanochat (see train.yml)
# Locally, attached:  HF_TOKEN=hf_... modal run modal_train.py --depth 12 --hf-repo yourname/nanochat
"""

import csv
import os
import re
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
# Persistent storage: dataset shards + tokenizer + checkpoints + logs
# ---------------------------------------------------------------------------
ckpt_vol = modal.Volume.from_name("nanochat-cache", create_if_missing=True, version=2)

# ---------------------------------------------------------------------------
# HF token: read from the environment of whoever launches the deploy
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

# ---------------------------------------------------------------------------
# Training-log parsing (stdlib only). Matches nanochat's step lines, e.g.:
# step 00180/02520 (7.14%) | loss: 3.790014 | lrm: 1.00 | dt: 991.36ms |
#   tok/sec: 528,855 | bf16_mfu: 40.62 | ...
# Non-matching lines (evals, warnings) are skipped; the raw log keeps them.
# ---------------------------------------------------------------------------
STEP_RE = re.compile(
    r"step (\d+)/(\d+) \(([\d.]+)%\) \| loss: ([\d.]+) \| lrm: ([\d.]+) \| "
    r"dt: ([\d.]+)ms \| tok/sec: ([\d,]+) \| bf16_mfu: ([\d.]+)"
)
CSV_FIELDS = ["step", "total_steps", "loss", "lrm", "dt_ms", "tok_per_sec", "bf16_mfu"]


def parse_log_to_csv(log_path: Path, csv_path: Path) -> int:
    """Extract per-step metrics from the raw training log into a CSV."""
    if not log_path.exists():
        return 0
    rows = []
    for line in log_path.read_text().splitlines():
        m = STEP_RE.search(line)
        if m:
            rows.append(
                {
                    "step": int(m.group(1)),
                    "total_steps": int(m.group(2)),
                    "loss": float(m.group(4)),
                    "lrm": float(m.group(5)),
                    "dt_ms": float(m.group(6)),
                    "tok_per_sec": int(m.group(7).replace(",", "")),
                    "bf16_mfu": float(m.group(8)),
                }
            )
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _periodic_volume_commit(stop: threading.Event, every_s: int = 900):
    """Flush Volume writes every 15 min so checkpoints/logs survive a crash."""
    while not stop.wait(every_s):
        try:
            ckpt_vol.commit()
            print("[modal_train] volume checkpoint committed")
        except Exception as e:  # never kill training over a commit hiccup
            print(f"[modal_train] volume commit failed (non-fatal): {e}")


def _run_streamed(cmd: str, log_path: Path | None = None):
    """Run cmd, streaming merged stdout+stderr into Modal logs, optionally
    teeing every line to log_path. Guarantees child tracebacks are visible."""
    print(f"[modal_train] running: {cmd}", flush=True)
    log_f = open(log_path, "a") if log_path else None
    try:
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
            if log_f:
                log_f.write(line)
        proc.wait()
    finally:
        if log_f:
            log_f.close()
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
    window_pattern: str = "L",
    data_shards: int = 32,
    extra_args: str = "",
) -> dict:
    """
    Pretrain nanochat at a given depth, checkpoint to the Volume, push
    checkpoints + training log + metrics CSV to HF.

    depth          : transformer depth (the one nanochat complexity dial)
    hf_repo        : HF repo id e.g. "yourname/nanochat"; empty skips upload
    save_every     : checkpoint interval in steps (persisted to the Volume)
    window_pattern : attention window pattern; "L" = full causal context
                     (our experiment default), "SSSL" = upstream sliding-window
    data_shards    : pretraining shards to ensure downloaded (d12 with 8 shards
                     repeated data 4x -> use ~32+ so epoch stays ~1; bigger
                     depths need more, see runs/speedrun.sh)
    extra_args     : extra CLI flags, e.g. "--device-batch-size=8"
    """
    os.chdir(REPO_DIR)
    run_name = f"d{depth}"

    # Log locations (on the Volume, so they persist and survive crashes)
    logs_dir = Path(CACHE_DIR) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{run_name}_train.log"
    csv_path = logs_dir / f"{run_name}_metrics.csv"
    log_path.unlink(missing_ok=True)  # fresh log per run

    # Background committer so mid-run checkpoints/logs are durable
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

    # Data shards: always top up to data_shards (incremental — already-present
    # shards are skipped; persisted on the Volume). The first run with only 8
    # shards produced 4 epochs of data repetition for d12; avoid that.
    _run_streamed(f"{python} -m nanochat.dataset -n {data_shards}")

    # Tokenizer: one-time (persisted on the Volume).
    # NOTE: mirror the exact commands from your fork's runs/speedrun.sh if
    # these ever drift from upstream.
    tokenizer_dir = Path(CACHE_DIR) / "tokenizer"
    if not tokenizer_dir.exists():
        print("[modal_train] no tokenizer found — training tokenizer (one-time)")
        _run_streamed(f"{python} -m scripts.tok_train")
    ckpt_vol.commit()

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
        f"--window-pattern={window_pattern} "
        f"{extra_args}"
    )
    t0 = time.time()
    try:
        _run_streamed(cmd, log_path=log_path)
    finally:
        stop.set()
        n = parse_log_to_csv(log_path, csv_path)
        print(f"[modal_train] parsed {n} step lines -> {csv_path}")
        ckpt_vol.commit()  # always flush whatever we have, even on failure
    print(f"[modal_train] training finished in {(time.time() - t0) / 3600:.2f} h")

    # ── Push checkpoints + logs to HuggingFace ───────────────────────────────
    if hf_repo:
        from huggingface_hub import HfApi

        token = os.environ.get("HF_TOKEN", "")
        if not token:
            print("[modal_train] WARNING: HF_TOKEN empty, skipping upload")
        else:
            # nanochat saves under ~/.cache/nanochat/base_checkpoints/<model-tag>/
            ckpt_dir = Path(CACHE_DIR) / "base_checkpoints" / run_name
            api = HfApi(token=token)
            api.create_repo(repo_id=hf_repo, private=True, exist_ok=True)
            if ckpt_dir.exists():
                api.upload_folder(
                    repo_id=hf_repo,
                    folder_path=str(ckpt_dir),
                    path_in_repo=f"checkpoints/{run_name}",
                )
            else:
                print(f"[modal_train] WARNING: no checkpoint dir at {ckpt_dir}, "
                      "skipping checkpoint upload (logs still uploaded)")
            for p, dest in [
                (csv_path, f"logs/{run_name}_metrics.csv"),
                (log_path, f"logs/{run_name}_train.log"),
            ]:
                if p.exists():
                    api.upload_file(
                        path_or_fileobj=str(p), path_in_repo=dest, repo_id=hf_repo
                    )
            print(f"[modal_train] pushed checkpoints + logs to {hf_repo}")

    return {"ok": True, "run": run_name, "hf_repo": hf_repo}


@app.local_entrypoint()
def main(
    depth: int = 12,
    hf_repo: str = "",
    save_every: int = 1000,
    window_pattern: str = "L",
    data_shards: int = 32,
    extra_args: str = "",
):
    train_nanochat.remote(
        depth=depth,
        hf_repo=hf_repo,
        save_every=save_every,
        window_pattern=window_pattern,
        data_shards=data_shards,
        extra_args=extra_args,
    )
