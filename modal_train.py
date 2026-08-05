"""
modal_train.py — train nanochat variants on Modal (1x H100), launched from GitHub Actions.

Lives in the repo root of your nanochat fork.

HF repo layout (hf_repo = sparsetrace/nanochat), one subfolder per attention
variant (hf_subfolder = "attention" | "AMAP" | "DMAP" | "Witten" | ...):

    <subfolder>/latest/                  # live checkpoint of the CURRENT run,
                                         #   overwritten every save_every steps
                                         #   (model + meta + optim: resume-capable)
    <subfolder>/checkpoints/<run_tag>/   # final archive per run (model + meta)
    <subfolder>/logs/<run_tag>_metrics.csv
    <subfolder>/logs/<run_tag>_train.log
    <subfolder>/samples/<run_tag>_samples.txt   # generated text from the run

Resume: before training we check <subfolder>/latest/ on HF. If found, its
files are downloaded into the local checkpoint dir and reported loudly.
NOTE: stock nanochat base_train has NO resume flag — it always trains from
step 0. The download+detection side is fully wired here; to actually continue
training, add a resume flag to your fork's base_train.py and pass it via
extra_args (grep this file for RESUME to find the hook point).

Samples: nanochat prints generation samples (lines starting with <|bos|>)
during/after training; we extract them from the teed log into samples/*.txt.
No separate inference harness needed — robust to model-code changes (works
for HMAP, which lacks kv-cache inference support).
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

HF_REPO_DEFAULT = "sparsetrace/nanochat"

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
        copy=True,
        ignore=[".git", ".venv", "__pycache__", "*.pyc", ".github"],
    )
    .run_commands(f"cd {REPO_DIR} && uv sync --extra gpu")
)

# ---------------------------------------------------------------------------
# Training-log parsing (stdlib only).
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


def extract_samples(log_path: Path, samples_path: Path, run_tag: str) -> int:
    """Pull nanochat's generation samples (<|bos|>... lines) out of the log."""
    if not log_path.exists():
        return 0
    lines = [l for l in log_path.read_text().splitlines() if l.startswith("<|bos|>")]
    with open(samples_path, "w") as f:
        f.write(f"# generation samples from run {run_tag}\n")
        if lines:
            f.write("\n".join(lines) + "\n")
        else:
            f.write("(no samples found in the training log for this run)\n")
    return len(lines)


def _periodic_sync(stop: threading.Event, api, hf_repo: str, subfolder: str,
                   local_ckpt_dir: Path, every_s: int = 900):
    """Every 15 min: commit the Volume AND mirror the newest checkpoint files
    to HF <subfolder>/latest/ (overwriting). Failures are logged, never fatal."""
    pushed_mtimes = {}
    while not stop.wait(every_s):
        try:
            ckpt_vol.commit()
            print("[modal_train] volume checkpoint committed")
        except Exception as e:
            print(f"[modal_train] volume commit failed (non-fatal): {e}")
        if api is None or not local_ckpt_dir.exists():
            continue
        try:
            for p in sorted(local_ckpt_dir.iterdir()):
                if not p.is_file():
                    continue
                mtime = p.stat().st_mtime
                if pushed_mtimes.get(p.name) == mtime:
                    continue  # unchanged since last push
                api.upload_file(
                    path_or_fileobj=str(p),
                    path_in_repo=f"{subfolder}/latest/{p.name}",
                    repo_id=hf_repo,
                )
                pushed_mtimes[p.name] = mtime
                print(f"[modal_train] mirrored {p.name} -> {subfolder}/latest/")
        except Exception as e:
            print(f"[modal_train] HF mirror failed (non-fatal): {e}")


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
    memory=32768,
    timeout=24 * 60 * 60,   # Modal's max
    retries=0,
    volumes={CACHE_DIR: ckpt_vol},
    secrets=[hf_secret],
)
def train_nanochat(
    depth: int = 12,
    hf_repo: str = HF_REPO_DEFAULT,
    hf_subfolder: str = "attention",
    save_every: int = 1000,
    window_pattern: str = "L",
    data_shards: int = 32,
    run_tag: str = "",
    resume: bool = False,
    extra_args: str = "",
) -> dict:
    """
    Pretrain a nanochat attention variant; live-mirror checkpoints to HF.

    depth          : transformer depth (the one nanochat complexity dial)
    hf_repo        : HF repo id (default sparsetrace/nanochat); empty skips HF
    hf_subfolder   : variant subfolder in the HF repo: "attention", "AMAP",
                     "DMAP", "Witten", ...
    save_every     : checkpoint interval in steps (mirrored to HF each cycle)
    window_pattern : "L" = full causal context (experiment default)
    data_shards    : pretraining shards to ensure downloaded (32+ for d12)
    run_tag        : distinct run name; defaults to "d{depth}". Use e.g.
                     "d12-a05" for the alpha=0.5 arm.
    resume         : if True, download <hf_subfolder>/latest/ from HF and pass
                     --resume-from-step to continue an interrupted run. The
                     run_tag, depth, and variant flags MUST match the original
                     run (the checkpoint carries model shapes and dataloader
                     state). Default False: fresh runs never silently continue
                     a finished predecessor.
    extra_args     : extra base_train flags, e.g.
                     "--attn-variant hmap --hmap-alpha 0.0 --device-batch-size=8"
    """
    os.chdir(REPO_DIR)
    run_name = run_tag if run_tag else f"d{depth}"

    # Local paths (on the Volume)
    ckpt_dir = Path(CACHE_DIR) / "base_checkpoints" / run_name
    logs_dir = Path(CACHE_DIR) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{run_name}_train.log"
    csv_path = logs_dir / f"{run_name}_metrics.csv"
    samples_path = logs_dir / f"{run_name}_samples.txt"
    log_path.unlink(missing_ok=True)  # fresh log per run

    # HF client (None disables all HF interaction)
    api = None
    token = os.environ.get("HF_TOKEN", "")
    if not hf_repo:
        print("[modal_train] hf_repo is EMPTY — all HF interaction disabled.")
    elif not token:
        print("[modal_train] HF_TOKEN is EMPTY — all HF interaction disabled.")
    else:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(repo_id=hf_repo, private=True, exist_ok=True)

    # ── Resume: look for <subfolder>/latest/ on HF ───────────────────────────
    # Opt-in (resume=True): download prior checkpoint files into ckpt_dir,
    # parse the step from meta_*.json filenames, and pass --resume-from-step
    # so base_train continues (model + optimizer + dataloader state).
    # Opt-out (default): report what exists but always train from scratch.
    resume_flag = ""
    if api is not None:
        try:
            existing = [
                f for f in api.list_repo_files(repo_id=hf_repo)
                if f.startswith(f"{hf_subfolder}/latest/")
            ]
            if existing and resume:
                from huggingface_hub import hf_hub_download
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                for repo_file in existing:
                    local = hf_hub_download(
                        repo_id=hf_repo, filename=repo_file, token=token,
                        local_dir=str(Path(CACHE_DIR) / "hf_resume"),
                    )
                    dest = ckpt_dir / Path(repo_file).name
                    dest.write_bytes(Path(local).read_bytes())
                # Parse resume step from meta_XXXXXX.json filenames (max = newest)
                steps = []
                for repo_file in existing:
                    m = re.match(r"meta_(\d+)\.json$", Path(repo_file).name)
                    if m:
                        steps.append(int(m.group(1)))
                if steps:
                    resume_step = max(steps)
                    resume_flag = f"--resume-from-step={resume_step} "
                    print(f"[modal_train] RESUME: continuing from step {resume_step} "
                          f"({len(existing)} file(s) from {hf_subfolder}/latest/ -> {ckpt_dir})")
                else:
                    print("[modal_train] RESUME requested but no meta_*.json found "
                          "in latest/ — cannot determine step, training fresh.")
            elif existing:
                print(f"[modal_train] {len(existing)} checkpoint file(s) exist in "
                      f"{hf_subfolder}/latest/ (resume=False — training fresh).")
            else:
                if resume:
                    print(f"[modal_train] RESUME requested but {hf_subfolder}/latest/ "
                          "is empty — training fresh.")
                else:
                    print(f"[modal_train] no prior checkpoint in {hf_subfolder}/latest/ "
                          "— fresh start.")
        except Exception as e:
            print(f"[modal_train] resume check failed (non-fatal, fresh start): {e}")

    # ── Data + tokenizer prerequisites (Volume-persistent) ──────────────────
    python = f"{VENV}/bin/python"
    torchrun = f"{VENV}/bin/torchrun"

    _run_streamed(
        f'{python} -c "import nanochat.gpt; '
        f"print('[sanity] gpt.py:', nanochat.gpt.__file__)\""
    )
    _run_streamed(f"{python} -m nanochat.dataset -n {data_shards}")
    tokenizer_dir = Path(CACHE_DIR) / "tokenizer"
    if not tokenizer_dir.exists():
        print("[modal_train] no tokenizer found — training tokenizer (one-time)")
        _run_streamed(f"{python} -m scripts.tok_train")
    ckpt_vol.commit()

    # ── Background: Volume commits + HF latest/ mirroring, every 15 min ─────
    stop = threading.Event()
    threading.Thread(
        target=_periodic_sync,
        args=(stop, api, hf_repo, hf_subfolder, ckpt_dir),
        daemon=True,
    ).start()

    # ── Train ────────────────────────────────────────────────────────────────
    if NPROC > 1:
        launcher = f"{torchrun} --standalone --nproc_per_node={NPROC} -m scripts.base_train --"
    else:
        launcher = f"{python} -m scripts.base_train"

    cmd = (
        f"{launcher} "
        f"--depth={depth} "
        f"--model-tag={run_name} "
        f"--save-every={save_every} "
        f"--window-pattern={window_pattern} "
        f"{resume_flag}"
        f"{extra_args}"
    )
    t0 = time.time()
    try:
        _run_streamed(cmd, log_path=log_path)
    finally:
        stop.set()
        n = parse_log_to_csv(log_path, csv_path)
        ns = extract_samples(log_path, samples_path, run_name)
        print(f"[modal_train] parsed {n} step lines -> {csv_path}; "
              f"extracted {ns} samples -> {samples_path}")
        ckpt_vol.commit()
    print(f"[modal_train] training finished in {(time.time() - t0) / 3600:.2f} h")

    # ── Final HF upload ──────────────────────────────────────────────────────
    if api is None:
        print("[modal_train] HF disabled — final artifacts remain on the Volume only.")
    else:
        # 1) latest/: full resume-capable set (model + meta + optim), overwrite
        if ckpt_dir.exists():
            for p in sorted(ckpt_dir.iterdir()):
                if p.is_file():
                    api.upload_file(path_or_fileobj=str(p),
                                    path_in_repo=f"{hf_subfolder}/latest/{p.name}",
                                    repo_id=hf_repo)
            # 2) per-run archive: model + meta only (skip big optimizer states)
            for p in sorted(ckpt_dir.iterdir()):
                if p.is_file() and not p.name.startswith("optim_"):
                    api.upload_file(
                        path_or_fileobj=str(p),
                        path_in_repo=f"{hf_subfolder}/checkpoints/{run_name}/{p.name}",
                        repo_id=hf_repo)
        else:
            print(f"[modal_train] WARNING: no checkpoint dir at {ckpt_dir}, "
                  "skipping checkpoint upload")
        # 3) logs + samples
        for p, dest in [
            (csv_path, f"{hf_subfolder}/logs/{run_name}_metrics.csv"),
            (log_path, f"{hf_subfolder}/logs/{run_name}_train.log"),
            (samples_path, f"{hf_subfolder}/samples/{run_name}_samples.txt"),
        ]:
            if p.exists():
                api.upload_file(path_or_fileobj=str(p), path_in_repo=dest,
                                repo_id=hf_repo)
        print(f"[modal_train] pushed checkpoints + logs + samples to "
              f"{hf_repo}/{hf_subfolder}/")

    return {"ok": True, "run": run_name, "hf_repo": hf_repo,
            "hf_subfolder": hf_subfolder}


@app.local_entrypoint()
def main(
    depth: int = 12,
    hf_repo: str = HF_REPO_DEFAULT,
    hf_subfolder: str = "attention",
    save_every: int = 1000,
    window_pattern: str = "L",
    data_shards: int = 32,
    run_tag: str = "",
    resume: bool = False,
    extra_args: str = "",
):
    train_nanochat.remote(
        depth=depth,
        hf_repo=hf_repo,
        hf_subfolder=hf_subfolder,
        save_every=save_every,
        window_pattern=window_pattern,
        data_shards=data_shards,
        run_tag=run_tag,
        resume=resume,
        extra_args=extra_args,
    )
