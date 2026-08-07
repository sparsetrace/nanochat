"""
modal_d32ft.py

AMAP reconditioning / continued-pretraining harness for Karpathy nanochat-d32.

FIRST launch, if sparsetrace/d32ft has no resume-capable checkpoint:
- install Karpathy's exact d32 tokenizer
- load karpathy/nanochat-d32:model_000650.pt as WEIGHTS ONLY
- instantiate this fork's depth-32 model with L attention + HMAP alpha=0 (AMAP)
- create a fresh optimizer and fresh base-data stream at local step 0

LATER launches:
- resume this experiment's model + optimizer + dataloader/loop state exactly

The checked-in scripts/base_train.py is not modified. At runtime we create an
ephemeral scripts/_d32ft_base_train.py that adds one --init-from-model argument.

VINTAGE NOTE (why the warm start is strict-with-allowlist, not strict=True):
the published checkpoint is Oct-2025 architecture. Its 1.9B params are fully
accounted for by matrices + wte/lm_head at vocab 65,536 — it has NO
value_embeds and none of the modern scalar/gate machinery (x0_lambdas,
resid_lambdas, smear_*, backout_lambda, ve_gate), which this fork's model has.
Missing keys are therefore expected, but ONLY on that allowlist; anything else
aborts. Vintage-absent parameters are set to GRAFT-NEUTRAL values (the source
model's implicit values: resid scaling 1.0, mixings 0.0, VE tables zeroed) so
the grafted model differs from the source by the AMAP operator alone.

IMPORTANT harness constraint: this function runs under the container's SYSTEM
python (modal + huggingface_hub only). torch exists ONLY inside the repo venv;
any torch work must happen in a subprocess under {VENV}/bin/python.
"""

import csv
import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import modal

app = modal.App("nanochat-d32ft")

GPU_CONFIG = "H100"  # overridden per invocation by Function.with_options(...)

HF_REPO_DEFAULT = "sparsetrace/d32ft"
SOURCE_REPO = "karpathy/nanochat-d32"
SOURCE_MODEL = "model_000650.pt"
SOURCE_META = "meta_000650.json"
SOURCE_TOKENIZER = ("tokenizer.pkl", "token_bytes.pt")

CACHE_DIR = "/root/.cache/nanochat-d32ft"
REPO_DIR = "/root/nanochat"
VENV = f"{REPO_DIR}/.venv"

ckpt_vol = modal.Volume.from_name(
    "nanochat-d32ft-cache", create_if_missing=True, version=2
)

hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})

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
            "HF_HUB_DISABLE_XET": "1",
            "NANOCHAT_BASE_DIR": CACHE_DIR,
            # Reduce fragmentation-induced OOM at borderline peak memory
            # (d32-eager sits at ~77-79GB of 80GB; allocation-order changes
            # across resume/compile-cache paths otherwise tip it over).
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            # Persist compile artifacts on the Volume: first d32-eager compile
            # costs many minutes; relaunches then reuse it instead of repaying.
            "TORCHINDUCTOR_CACHE_DIR": f"{CACHE_DIR}/inductor_cache",
            "TRITON_CACHE_DIR": f"{CACHE_DIR}/triton_cache",
        }
    )
    .add_local_dir(
        ".",
        remote_path=REPO_DIR,
        copy=True,
        ignore=[
            ".git", ".venv", "__pycache__", "*.pyc", ".github",
            "modal_train.py", "modal_d32ft.py", "modal_sample.py", "modal_eval.py",
        ],
    )
    .run_commands(f"cd {REPO_DIR} && uv sync --extra gpu")
)

STEP_RE = re.compile(
    r"step (\d+)/(\d+) \(([\d.]+)%\) \| loss: ([\d.]+) \| lrm: ([\d.]+) \| "
    r"dt: ([\d.]+)ms \| tok/sec: ([\d,]+) \| bf16_mfu: ([\d.]+)"
)
CSV_FIELDS = ["step", "total_steps", "loss", "lrm", "dt_ms", "tok_per_sec", "bf16_mfu"]

ICL_RE = re.compile(
    r"Step (\d+) \| ICL early: ([\d.]+) late: ([\d.]+) score: ([+\-][\d.]+) \| "
    r"Induction loss: ([\d.]+) acc: ([\d.]+) \(random-half: ([\d.]+)\)"
)
ICL_FIELDS = [
    "step", "icl_early", "icl_late", "icl_score",
    "induction_loss", "induction_acc", "random_half_loss",
]


def _run_streamed(cmd: str, log_path: Path | None = None):
    print(f"[d32ft] running: {cmd}", flush=True)
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


def _count_visible_gpus() -> int:
    """GPU probe via the repo venv. The harness process itself has no torch
    (system python), so `import torch` here would ModuleNotFoundError — all
    torch work belongs in subprocesses under the venv interpreter."""
    probe = subprocess.run(
        [f"{VENV}/bin/python", "-c",
         "import torch; print(torch.cuda.device_count())"],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        raise RuntimeError(f"GPU probe failed: {probe.stderr.strip()}")
    return int(probe.stdout.strip())


def parse_log_to_csv(log_path: Path, csv_path: Path) -> int:
    rows = []
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            m = STEP_RE.search(line)
            if m:
                rows.append({
                    "step": int(m.group(1)),
                    "total_steps": int(m.group(2)),
                    "loss": float(m.group(4)),
                    "lrm": float(m.group(5)),
                    "dt_ms": float(m.group(6)),
                    "tok_per_sec": int(m.group(7).replace(",", "")),
                    "bf16_mfu": float(m.group(8)),
                })
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def parse_icl_to_csv(log_path: Path, csv_path: Path) -> int:
    rows = []
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            m = ICL_RE.search(line)
            if m:
                rows.append({
                    "step": int(m.group(1)),
                    "icl_early": float(m.group(2)),
                    "icl_late": float(m.group(3)),
                    "icl_score": float(m.group(4)),
                    "induction_loss": float(m.group(5)),
                    "induction_acc": float(m.group(6)),
                    "random_half_loss": float(m.group(7)),
                })
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ICL_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def extract_samples(log_path: Path, samples_path: Path, run_tag: str) -> int:
    lines = []
    if log_path.exists():
        lines = [
            line for line in log_path.read_text().splitlines()
            if line.startswith("<|bos|>")
        ]
    with open(samples_path, "w") as f:
        f.write(f"# generation samples from run {run_tag}\n")
        if lines:
            f.write("\n".join(lines) + "\n")
    return len(lines)


def _install_warmstart_train_module() -> str:
    src = Path(REPO_DIR) / "scripts" / "base_train.py"
    dst = Path(REPO_DIR) / "scripts" / "_d32ft_base_train.py"
    text = src.read_text()

    arg_anchor = (
        'parser.add_argument("--resume-from-step", type=int, default=-1, '
        'help="resume training from this step (-1 = disable)")'
    )
    if arg_anchor not in text:
        raise RuntimeError(
            "Could not patch base_train.py: --resume-from-step anchor changed."
        )
    text = text.replace(
        arg_anchor,
        arg_anchor
        + '\nparser.add_argument("--init-from-model", type=str, default="", '
          'help="weights-only warm start from a raw nanochat state_dict .pt")',
        1,
    )

    load_anchor = "    del model_data # free up this memory after the copy\n"
    if load_anchor not in text:
        raise RuntimeError(
            "Could not patch base_train.py: checkpoint load anchor changed."
        )

    # Strict-with-allowlist warm start (see VINTAGE NOTE in module docstring):
    #  - unexpected keys        -> hard error (checkpoint has things we lack)
    #  - shape mismatches       -> hard error (wrong tokenizer/config)
    #  - missing keys           -> hard error UNLESS on the vintage allowlist
    #  - allowlisted missing    -> set to GRAFT-NEUTRAL values (see below)
    warmstart = r"""
elif args.init_from_model:
    print0(f"[d32ft] Weights-only warm start from {args.init_from_model}")
    print0("[d32ft] New optimizer + new dataloader state; local step starts at 0")
    _init_state = torch.load(
        args.init_from_model,
        map_location=device,
        weights_only=True,
    )
    _own = model.state_dict()
    _missing = sorted(set(_own) - set(_init_state))
    _unexpected = sorted(set(_init_state) - set(_own))
    for _k in _missing:
        print0(f"[d32ft]   missing from checkpoint: {_k}")
    for _k in _unexpected:
        print0(f"[d32ft]   unexpected in checkpoint: {_k}")
    _shape_bad = [
        (_k, tuple(_init_state[_k].shape), tuple(_own[_k].shape))
        for _k in set(_own) & set(_init_state)
        if tuple(_init_state[_k].shape) != tuple(_own[_k].shape)
    ]
    for _k, _cs, _ms in _shape_bad:
        print0(f"[d32ft]   SHAPE MISMATCH: {_k} ckpt{_cs} vs model{_ms}")
    assert not _shape_bad, \
        "[d32ft] shape mismatch — wrong tokenizer or model config for this checkpoint"
    assert not _unexpected, \
        "[d32ft] checkpoint contains keys this model lacks — config mismatch"
    _ALLOW = ("value_embed", "ve_gate", "resid_lambda", "x0_lambda",
              "smear_gate", "smear_lambda", "backout_lambda")
    _bad_missing = [
        _k for _k in _missing if not any(_a in _k for _a in _ALLOW)
    ]
    assert not _bad_missing, \
        f"[d32ft] missing keys outside the vintage allowlist: {_bad_missing}"
    model.load_state_dict(_init_state, strict=False)
    # Policy for vintage-absent modules — GRAFT-NEUTRAL values, i.e. the
    # source model's implicit values, NOT init_weights' from-scratch values:
    #  - value_embeds TABLES: zero (additive content; zero = no contribution).
    #  - ve_gate / smear_gate: keep init (small, and gated by zeroed
    #    tables / smear_lambda=0 respectively — inert at step 0).
    #  - resid_lambdas -> 1.0 exactly. init_weights gives ~1.13-1.15, which
    #    compounds ~50x over 32 layers on a stream trained at implicit 1.0.
    #  - x0_lambdas -> 0.0 (source model had no x0 mixing).
    #  - backout_lambda -> 0.0 (source model had no backout mixing).
    #  - smear_lambda: init is already 0.0 (verified in log) — kept.
    # The step-0 val bpb is the ultimate adjudicator of neutrality.
    with torch.no_grad():
        for _k, _p in model.named_parameters():
            if _k not in _missing:
                continue
            if "value_embeds" in _k:
                _p.zero_()
                print0(f"[d32ft]   zero-initialized VE table: {_k}")
            elif "resid_lambda" in _k:
                _p.fill_(1.0)
                print0(f"[d32ft]   graft-neutral override: {_k} -> 1.0 "
                       f"(was from-scratch init ~1.13)")
            elif "x0_lambda" in _k or "backout_lambda" in _k:
                _p.zero_()
                print0(f"[d32ft]   graft-neutral override: {_k} -> 0.0 "
                       f"(was from-scratch init)")
            else:
                _flat = _p.detach().float().flatten()
                _preview = ", ".join(f"{v:.3f}" for v in _flat[:8].tolist())
                _suffix = ", ..." if _flat.numel() > 8 else ""
                print0(f"[d32ft]   kept init (verify neutral!): {_k} "
                       f"shape={tuple(_p.shape)} "
                       f"mean={_flat.mean().item():.4f} "
                       f"values=[{_preview}{_suffix}]")
    del _init_state, _own
"""
    text = text.replace(load_anchor, load_anchor + warmstart, 1)
    dst.write_text(text)
    print(f"[d32ft] installed ephemeral trainer: {dst}")
    return "scripts._d32ft_base_train"


def _hf_download_into(repo_id: str, filename: str, dest: Path, token=None):
    from huggingface_hub import hf_hub_download
    cached = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=token,
        local_dir=str(Path(CACHE_DIR) / "hf_downloads" / repo_id.replace("/", "__")),
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, dest)
    return dest


def _ensure_source_tokenizer():
    tok_dir = Path(CACHE_DIR) / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_TOKENIZER:
        _hf_download_into(SOURCE_REPO, name, tok_dir / name, token=None)
    print(f"[d32ft] installed exact Karpathy d32 tokenizer -> {tok_dir}")


def _download_karpathy_seed() -> Path:
    seed_dir = Path(CACHE_DIR) / "bootstrap" / "karpathy-d32"
    model_path = _hf_download_into(
        SOURCE_REPO, SOURCE_MODEL, seed_dir / SOURCE_MODEL, token=None
    )
    _hf_download_into(
        SOURCE_REPO, SOURCE_META, seed_dir / SOURCE_META, token=None
    )
    print(f"[d32ft] source seed ready: {model_path}")
    return model_path


def _checkpoint_steps(files: list[str], prefix: str) -> list[int]:
    model_steps, meta_steps, optim_steps = set(), set(), set()
    for f in files:
        if not f.startswith(prefix):
            continue
        name = Path(f).name
        m = re.fullmatch(r"model_(\d+)\.pt", name)
        if m:
            model_steps.add(int(m.group(1)))
        m = re.fullmatch(r"meta_(\d+)\.json", name)
        if m:
            meta_steps.add(int(m.group(1)))
        # DDP runs save per-rank optimizer files (optim_000100_rank0.pt ...);
        # single-GPU runs save optim_000100.pt. Accept both.
        m = re.fullmatch(r"optim_(\d+)(?:_rank\d+)?\.pt", name)
        if m:
            optim_steps.add(int(m.group(1)))
    return sorted(model_steps & meta_steps & optim_steps)


def _restore_own_checkpoint(api, hf_repo: str, token: str, run_name: str):
    from huggingface_hub import hf_hub_download

    ckpt_dir = Path(CACHE_DIR) / "base_checkpoints" / run_name

    # ── Volume first: the previous run SAVED here; reading it back costs
    #    seconds. HF is the durable mirror, not the hot path — paying
    #    8xH100 rates to re-download ~30GB we already have is silly. ──────────
    local_names = ([p.name for p in ckpt_dir.iterdir() if p.is_file()]
                   if ckpt_dir.exists() else [])
    local_steps = _checkpoint_steps(local_names, "")
    best_local = local_steps[-1] if local_steps else -1

    files = list(api.list_repo_files(repo_id=hf_repo))
    candidates = [
        ("latest/", _checkpoint_steps(files, "latest/")),
        (
            f"checkpoints/{run_name}/",
            _checkpoint_steps(files, f"checkpoints/{run_name}/"),
        ),
    ]
    chosen_prefix, chosen_step = None, -1
    for prefix, steps in candidates:
        if steps and steps[-1] > chosen_step:
            chosen_prefix, chosen_step = prefix, steps[-1]

    if best_local < 0 and chosen_prefix is None:
        return None

    if best_local >= chosen_step:
        # Volume has the newest (or only) checkpoint — no download at all.
        use_step = best_local
        print(
            f"[d32ft] RESUME (volume, no download): step {use_step} "
            f"already in {ckpt_dir}"
        )
    else:
        # HF is newer (e.g. fresh volume, or another workspace trained since).
        use_step = chosen_step
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        step_token = f"{use_step:06d}"
        print(
            f"[d32ft] RESUME (HF download): {hf_repo}/{chosen_prefix} "
            f"step {use_step} (volume has {best_local})"
        )
        for repo_file in files:
            name = Path(repo_file).name
            if repo_file.startswith(chosen_prefix) and step_token in name and (
                name.startswith("model_") or name.startswith("meta_")
                or name.startswith("optim_")
            ):
                cached = hf_hub_download(
                    repo_id=hf_repo,
                    filename=repo_file,
                    token=token,
                    # /tmp = fast local disk; staging through the Volume
                    # doubled every byte over the network FS.
                    local_dir="/tmp/hf_resume",
                )
                shutil.copy2(cached, ckpt_dir / name)

    meta = json.loads((ckpt_dir / f"meta_{use_step:06d}.json").read_text())
    uc = meta.get("user_config", {}) or {}
    if int(uc.get("depth", -1)) != 32:
        raise RuntimeError(
            f"Refusing non-d32 checkpoint: depth={uc.get('depth')}"
        )
    if uc.get("model_tag") not in (None, run_name):
        raise RuntimeError(
            f"Refusing model_tag={uc.get('model_tag')!r}; expected {run_name!r}"
        )

    print(f"[d32ft] RESUME: step {use_step} -> {ckpt_dir}")
    return use_step, ckpt_dir


def _periodic_sync(
    stop: threading.Event,
    api,
    hf_repo: str,
    local_ckpt_dir: Path,
    every_s: int = 900,
):
    from huggingface_hub import CommitOperationAdd
    pushed_mtimes = {}

    while not stop.wait(every_s):
        try:
            ckpt_vol.commit()
            print("[d32ft] volume committed")
        except Exception as e:
            print(f"[d32ft] volume commit failed (non-fatal): {e}")

        if api is None or not local_ckpt_dir.exists():
            continue

        try:
            ops, names = [], []
            for p in sorted(local_ckpt_dir.iterdir()):
                if not p.is_file():
                    continue
                mtime = p.stat().st_mtime
                if pushed_mtimes.get(p.name) == mtime:
                    continue
                ops.append(
                    CommitOperationAdd(
                        path_in_repo=f"latest/{p.name}",
                        path_or_fileobj=str(p),
                    )
                )
                names.append((p.name, mtime))

            if stop.is_set():
                print("[d32ft] mirror cycle skipped (training finished)")
                break

            if ops:
                api.create_commit(
                    repo_id=hf_repo,
                    operations=ops,
                    commit_message=f"d32ft mirror latest ({len(ops)} files)",
                )
                for name, mtime in names:
                    pushed_mtimes[name] = mtime
                print(f"[d32ft] mirrored {len(ops)} file(s) -> latest/")
        except Exception as e:
            print(f"[d32ft] HF mirror failed (non-fatal): {e}")


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=4 * 60 * 60,
    scaledown_window=5,
    volumes={CACHE_DIR: ckpt_vol},
    secrets=[hf_secret],
)
def upload_artifacts(
    hf_repo: str,
    run_tag: str,
    horizon: int,
    keep_last_steps: int = 1,
) -> dict:
    """Detached CPU stage-out: pushes checkpoint + logs to HF and prunes
    superseded checkpoint steps from the Volume. Spawned by train_d32ft at
    the end of a run so the 8xH100 container never idles through a ~30GB
    upload (CPU worker: cents; idle GPUs: ~$5-10 per upload)."""
    from huggingface_hub import HfApi, CommitOperationAdd

    token = os.environ.get("HF_TOKEN", "")
    api = HfApi(token=token)
    ckpt_vol.reload()  # see the training container's committed writes

    ckpt_dir = Path(CACHE_DIR) / "base_checkpoints" / run_tag
    logs_dir = Path(CACHE_DIR) / "logs"

    ops = []
    if ckpt_dir.exists():
        for p in sorted(ckpt_dir.iterdir()):
            if not p.is_file():
                continue
            ops.append(CommitOperationAdd(
                path_in_repo=f"latest/{p.name}", path_or_fileobj=str(p)))
            ops.append(CommitOperationAdd(
                path_in_repo=f"checkpoints/{run_tag}/{p.name}",
                path_or_fileobj=str(p)))
    for name, dest in [
        (f"{run_tag}_metrics.csv", f"logs/{run_tag}_metrics.csv"),
        (f"{run_tag}_icl.csv", f"logs/{run_tag}_icl.csv"),
        (f"{run_tag}_train.log", f"logs/{run_tag}_train.log"),
        (f"{run_tag}_samples.txt", f"samples/{run_tag}_samples.txt"),
        (f"{run_tag}_provenance.json", f"logs/{run_tag}_provenance.json"),
    ]:
        p = logs_dir / name
        if p.exists():
            ops.append(CommitOperationAdd(path_in_repo=dest, path_or_fileobj=str(p)))
    if ops:
        api.create_commit(
            repo_id=hf_repo,
            operations=ops,
            commit_message=f"{run_tag}: AMAP d32ft through step {horizon}",
        )
        print(f"[d32ft-upload] pushed {len(ops)} file(s) in one commit")

    # Prune superseded checkpoint steps from the Volume (keep newest N).
    # NOTE: prunes ONLY base_checkpoints/<run_tag>/ step files — the
    # tokenizer, dataset shards, Karpathy seed, and compile caches stay:
    # they are the warm state that makes relaunches fast.
    if ckpt_dir.exists():
        step_re = re.compile(r"(?:model|meta|optim)_(\d+)")
        steps = sorted({int(m.group(1)) for p in ckpt_dir.iterdir()
                        if (m := step_re.match(p.name))})
        for old in steps[:-keep_last_steps] if keep_last_steps > 0 else []:
            tok = f"{old:06d}"
            for p in list(ckpt_dir.iterdir()):
                if tok in p.name:
                    p.unlink()
            print(f"[d32ft-upload] pruned superseded step {old} from Volume")
        ckpt_vol.commit()

    print("[d32ft-upload] all done, returning.")
    return {"ok": True, "run": run_tag, "uploaded": len(ops)}


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    cpu=32,
    memory=131072,
    timeout=24 * 60 * 60,
    retries=0,
    scaledown_window=5,
    volumes={CACHE_DIR: ckpt_vol},
    secrets=[hf_secret],
)
def train_d32ft(
    hf_repo: str = HF_REPO_DEFAULT,
    run_tag: str = "d32ft-amap",
    num_gpus: int = 1,
    resume: str = "auto",
    add_steps: int = 1000,
    save_every: int = 250,
    data_shards: int = 240,
    extra_args: str = "",
) -> dict:
    if add_steps <= 0:
        raise ValueError("d32ft requires add_steps > 0")
    if resume not in {"auto", "never", "force"}:
        raise ValueError("resume must be auto|never|force")
    if num_gpus not in {1, 2, 4, 8}:
        raise ValueError("num_gpus must be one of 1, 2, 4, 8")

    os.chdir(REPO_DIR)

    # GPU sanity via the venv (the harness process has no torch — see docstring)
    visible_gpus = _count_visible_gpus()
    print(
        f"[d32ft] GPU sanity: requested={num_gpus}, "
        f"visible={visible_gpus}, CUDA_VISIBLE_DEVICES="
        f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
        flush=True,
    )
    if visible_gpus != num_gpus:
        raise RuntimeError(
            f"Modal GPU provisioning mismatch: requested {num_gpus} GPU(s), "
            f"but torch sees {visible_gpus}. Refusing to launch torchrun with "
            f"an invalid local world size."
        )
    nproc = visible_gpus
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN is required for sparsetrace/d32ft")

    from huggingface_hub import HfApi, CommitOperationAdd

    api = HfApi(token=token)
    api.create_repo(repo_id=hf_repo, private=True, exist_ok=True)

    # The published d32 embedding/unembedding rows correspond to this tokenizer.
    print("[d32ft] stage: installing exact Karpathy d32 tokenizer...", flush=True)
    _ensure_source_tokenizer()
    print("[d32ft] stage complete: tokenizer ready", flush=True)

    ckpt_dir = Path(CACHE_DIR) / "base_checkpoints" / run_tag
    logs_dir = Path(CACHE_DIR) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{run_tag}_train.log"
    csv_path = logs_dir / f"{run_tag}_metrics.csv"
    icl_csv_path = logs_dir / f"{run_tag}_icl.csv"
    samples_path = logs_dir / f"{run_tag}_samples.txt"
    log_path.unlink(missing_ok=True)

    resumed_step = 0
    resume_flag = ""
    init_flag = ""
    origin = ""

    restored = None
    if resume != "never":
        print(
            f"[d32ft] stage: checking {hf_repo} for a resume-capable checkpoint...",
            flush=True,
        )
        restored = _restore_own_checkpoint(api, hf_repo, token, run_tag)
        if restored is None:
            print("[d32ft] stage complete: no own checkpoint found", flush=True)
        else:
            print("[d32ft] stage complete: own checkpoint found", flush=True)
    else:
        print("[d32ft] resume='never' — skipping own-checkpoint lookup", flush=True)

    if restored is not None:
        resumed_step, ckpt_dir = restored
        resume_flag = f"--resume-from-step={resumed_step} "
        origin = f"{hf_repo}:step-{resumed_step}"
    else:
        if resume == "force":
            raise RuntimeError(
                f"resume='force' but no resume-capable checkpoint exists in {hf_repo}"
            )
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[d32ft] stage: downloading Karpathy d32 seed ({SOURCE_MODEL})...",
            flush=True,
        )
        seed_model = _download_karpathy_seed()
        print("[d32ft] stage complete: Karpathy d32 seed ready", flush=True)
        init_flag = f"--init-from-model={seed_model} "
        origin = f"{SOURCE_REPO}:{SOURCE_MODEL}"
        print(
            "[d32ft] BOOTSTRAP: public d32 weights -> AMAP operator; "
            "optimizer/dataloader start fresh at step 0"
        )

    python = f"{VENV}/bin/python"
    torchrun = f"{VENV}/bin/torchrun"

    # Historical d32 used 240 downloaded shards. This invokes your fork's
    # current nanochat.dataset implementation; pin the historical nanochat
    # commit if you need byte-for-byte reproduction of the 2025 corpus.
    print(
        f"[d32ft] stage: ensuring {data_shards} nanochat dataset shards are available...",
        flush=True,
    )
    _run_streamed(f"{python} -m nanochat.dataset -n {data_shards}")
    print("[d32ft] stage complete: dataset ready", flush=True)

    print(
        "[d32ft] stage: installing ephemeral weights-only warmstart trainer...",
        flush=True,
    )
    trainer_module = _install_warmstart_train_module()
    print(
        f"[d32ft] stage complete: trainer module = {trainer_module}",
        flush=True,
    )

    stop = threading.Event()
    sync_thread = threading.Thread(
        target=_periodic_sync,
        args=(stop, api, hf_repo, ckpt_dir),
        daemon=True,
    )
    sync_thread.start()

    if nproc > 1:
        launcher = (
            f"{torchrun} --standalone --nproc_per_node={nproc} "
            f"-m {trainer_module} --"
        )
    else:
        launcher = f"{python} -m {trainer_module}"

    horizon = resumed_step + add_steps
    cmd = (
        f"{launcher} "
        f"--depth=32 "
        f"--model-tag={run_tag} "
        f"--save-every={save_every} "
        f"--window-pattern=L "
        f"--num-iterations={horizon} "
        f"{resume_flag}"
        f"{init_flag}"
        f"{extra_args}"
    )

    print(f"[d32ft] origin: {origin}", flush=True)
    print(
        f"[d32ft] additive budget: {add_steps} new steps "
        f"({resumed_step} -> {horizon})",
        flush=True,
    )
    print(
        "[d32ft] stage: launching depth-32 AMAP training process "
        "(model load / torch.compile can take a while)...",
        flush=True,
    )

    t0 = time.time()
    trained_rows = 0
    try:
        _run_streamed(cmd, log_path=log_path)
    finally:
        stop.set()
        if sync_thread.is_alive():
            print("[d32ft] waiting for background mirror cycle to finish...")
            sync_thread.join(timeout=600)
        trained_rows = parse_log_to_csv(log_path, csv_path)
        n_icl = parse_icl_to_csv(log_path, icl_csv_path)
        ns = extract_samples(log_path, samples_path, run_tag)
        print(f"[d32ft] parsed {trained_rows} step lines; {n_icl} ICL lines; "
              f"{ns} samples")
        ckpt_vol.commit()

    # Zero-step guard: a resume that had nothing left to do re-saves an
    # identical checkpoint; re-pushing ~20GB of d32 artifacts helps no one.
    # (Bootstrap runs always push: even at low step counts the grafted
    # checkpoint + step-0 eval logs are the point of the exercise.)
    if trained_rows == 0 and resume_flag:
        print("[d32ft] 0 training steps on a resume — skipping final upload.")
        print("[d32ft] all done, returning. (Anything after this line is "
              "platform teardown, not this script.)")
        return {
            "ok": True, "run": run_tag, "hf_repo": hf_repo, "origin": origin,
            "start_step": resumed_step, "end_step": resumed_step,
            "parsed_training_lines": 0,
        }

    provenance_path = logs_dir / f"{run_tag}_provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "run_tag": run_tag,
                "origin": origin,
                "source_repo": SOURCE_REPO,
                "source_model": SOURCE_MODEL,
                "attention": "HMAP alpha=0.0 (AMAP)",
                "window_pattern": "L",
                "depth": 32,
                "resumed_step": resumed_step,
                "horizon": horizon,
                "new_steps_requested": add_steps,
                "num_gpus": nproc,
                "data_shards_requested": data_shards,
                "extra_args": extra_args,
                "graft_neutralization": {
                    "value_embeds": "zeroed",
                    "resid_lambdas": 1.0,
                    "x0_lambdas": 0.0,
                    "backout_lambda": 0.0,
                    "smear_lambda": "init (0.0)",
                    "ve_gate/smear_gate": "init (inert via zeroed tables / smear_lambda=0)",
                },
            },
            indent=2,
        )
        + "\n"
    )

    # Archive optimizer too: both latest/ and checkpoints/<run_tag>/ are
    # independently resume-capable.
    ops = []
    if ckpt_dir.exists():
        for p in sorted(ckpt_dir.iterdir()):
            if not p.is_file():
                continue
            ops.append(
                CommitOperationAdd(
                    path_in_repo=f"latest/{p.name}",
                    path_or_fileobj=str(p),
                )
            )
            ops.append(
                CommitOperationAdd(
                    path_in_repo=f"checkpoints/{run_tag}/{p.name}",
                    path_or_fileobj=str(p),
                )
            )

    for p, dest in [
        (csv_path, f"logs/{run_tag}_metrics.csv"),
        (icl_csv_path, f"logs/{run_tag}_icl.csv"),
        (log_path, f"logs/{run_tag}_train.log"),
        (samples_path, f"samples/{run_tag}_samples.txt"),
        (provenance_path, f"logs/{run_tag}_provenance.json"),
    ]:
        if p.exists():
            ops.append(CommitOperationAdd(path_in_repo=dest, path_or_fileobj=str(p)))

    if ops:
        api.create_commit(
            repo_id=hf_repo,
            operations=ops,
            commit_message=f"{run_tag}: AMAP d32ft through step {horizon}",
        )

    elapsed_h = (time.time() - t0) / 3600
    print(f"[d32ft] finished in {elapsed_h:.2f} h")
    print("[d32ft] all done, returning. (Anything after this line is "
          "platform teardown, not this script.)")
    return {
        "ok": True,
        "run": run_tag,
        "hf_repo": hf_repo,
        "origin": origin,
        "start_step": resumed_step,
        "end_step": horizon,
        "parsed_training_lines": trained_rows,
    }


# ── CORE evaluation: original Karpathy d32 AND converted AMAP checkpoints ───
CORE_EVAL_PY = r'''
import json, sys, torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import compute_init

cfg = json.load(open(sys.argv[1]))
template = json.load(open(cfg["template_meta"]))["model_config"]
template["attn_variant"] = cfg["attn_variant"]
print(f"[d32-core] source={cfg['source']} attn_variant={cfg['attn_variant']}",
      flush=True)

ddp, rank, local_rank, world, device = compute_init("cuda")

model = GPT(GPTConfig(**template)).to(device)
state = torch.load(cfg["model_path"], map_location=device, weights_only=True)
own = model.state_dict()
missing = sorted(set(own) - set(state))
unexpected = sorted(set(state) - set(own))
assert not unexpected, f"unexpected keys: {unexpected[:5]}"
ALLOW = ("value_embed", "ve_gate", "resid_lambda", "x0_lambda",
         "smear_gate", "smear_lambda", "backout_lambda")
bad = [k for k in missing if not any(a in k for a in ALLOW)]
assert not bad, f"missing keys outside vintage allowlist: {bad[:5]}"
model.load_state_dict(state, strict=False)
with torch.no_grad():
    for k, p in model.named_parameters():
        if k not in missing:
            continue
        if "value_embeds" in k:
            p.zero_()
        elif "resid_lambda" in k:
            p.fill_(1.0)
        elif "x0_lambda" in k or "backout_lambda" in k:
            p.zero_()
model.eval()

tokenizer = get_tokenizer()
from scripts.base_eval import evaluate_core
core = evaluate_core(model, tokenizer, device,
                     max_per_task=cfg["max_per_task"])
result = {
    "source": cfg["source"],
    "attn_variant": cfg["attn_variant"],
    "core_metric": core["core_metric"],
    "centered_results": core["centered_results"],
}
print(f"[d32-core] CORE metric: {core['core_metric']:.4f}", flush=True)
with open(cfg["out_path"], "w") as f:
    json.dump(result, f, indent=2)
print(f"[d32-core] wrote {cfg['out_path']}", flush=True)
'''


@app.function(
    image=image,
    gpu="H200",
    cpu=8,
    memory=65536,
    timeout=4 * 60 * 60,
    scaledown_window=5,
    volumes={CACHE_DIR: ckpt_vol},
    secrets=[hf_secret],
)
def core_eval_d32(
    source: str = "converted",   # "converted" (AMAP) | "original" (Karpathy seed)
    hf_repo: str = HF_REPO_DEFAULT,
    run_tag: str = "d32ft-amap",
    step: int = -1,              # converted only: -1 = newest on Volume
    max_per_task: int = 500,
) -> dict:
    """CORE metric (22-task capability suite) on a fresh GPU. 'original' =
    Karpathy seed under standard attention (FA3, fast); 'converted' = AMAP
    checkpoint via the chunked eager eval path (slow: budget ~1-2h).
    Identical vintage handling to the trainer/probe/sampler, so all four
    tools measure the same pair of functions."""
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN", "")
    api = HfApi(token=token) if token else None
    _ensure_source_tokenizer()
    ckpt_vol.reload()

    ckpt_dir = Path(CACHE_DIR) / "base_checkpoints" / run_tag
    local_names = ([p.name for p in ckpt_dir.iterdir() if p.is_file()]
                   if ckpt_dir.exists() else [])
    steps = _checkpoint_steps(local_names, "")
    assert steps, f"no converted checkpoint on Volume under {ckpt_dir}"
    tmpl_step = step if (step > 0 and step in steps) else steps[-1]
    template_meta = ckpt_dir / f"meta_{tmpl_step:06d}.json"

    if source == "original":
        model_path = _download_karpathy_seed()
        attn_variant = "standard"
        label = "original"
    else:
        model_path = ckpt_dir / f"model_{tmpl_step:06d}.pt"
        attn_variant = "hmap"
        label = f"{run_tag}-step{tmpl_step}"

    workdir = Path("/tmp/d32_core")
    workdir.mkdir(parents=True, exist_ok=True)
    out_path = workdir / f"{label}_core.json"
    job = {
        "source": source, "attn_variant": attn_variant,
        "template_meta": str(template_meta), "model_path": str(model_path),
        "max_per_task": max_per_task, "out_path": str(out_path),
    }
    (workdir / "job.json").write_text(json.dumps(job))
    (workdir / "core_script.py").write_text(CORE_EVAL_PY)

    _run_streamed(
        f"cd {REPO_DIR} && PYTHONPATH={REPO_DIR} "
        f"{VENV}/bin/python -u {workdir}/core_script.py {workdir}/job.json"
    )

    if api is not None:
        dest = f"evals/{label}_core.json"
        api.upload_file(path_or_fileobj=str(out_path), path_in_repo=dest,
                        repo_id=hf_repo)
        print(f"[d32-core] uploaded -> {hf_repo}/{dest}")
    print("[d32-core] all done, returning.")
    return json.loads(out_path.read_text())


# ── Sampling: original Karpathy d32 AND converted AMAP checkpoints ──────────
SAMPLE_PY = r'''
import json, sys, torch
from pathlib import Path
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer

cfg = json.load(open(sys.argv[1]))
template = json.load(open(cfg["template_meta"]))["model_config"]
template["attn_variant"] = cfg["attn_variant"]
print(f"[d32-sample] source={cfg['source']} attn_variant={cfg['attn_variant']}",
      flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT(GPTConfig(**template)).to(device)
state = torch.load(cfg["model_path"], map_location=device, weights_only=True)
own = model.state_dict()
missing = sorted(set(own) - set(state))
unexpected = sorted(set(state) - set(own))
assert not unexpected, f"unexpected keys: {unexpected[:5]}"
ALLOW = ("value_embed", "ve_gate", "resid_lambda", "x0_lambda",
         "smear_gate", "smear_lambda", "backout_lambda")
bad = [k for k in missing if not any(a in k for a in ALLOW)]
assert not bad, f"missing keys outside vintage allowlist: {bad[:5]}"
model.load_state_dict(state, strict=False)
with torch.no_grad():
    for k, p in model.named_parameters():
        if k not in missing:
            continue
        if "value_embeds" in k:
            p.zero_()
        elif "resid_lambda" in k:
            p.fill_(1.0)
        elif "x0_lambda" in k or "backout_lambda" in k:
            p.zero_()
model.eval()
if missing:
    print(f"[d32-sample] graft-neutralized {len(missing)} vintage-absent params",
          flush=True)

tokenizer = get_tokenizer()
lines = []
for prompt in cfg["prompts"]:
    for i in range(cfg["num_samples"]):
        tokens = tokenizer(prompt, prepend="<|bos|>")
        with torch.no_grad():
            out = list(model.generate(
                tokens, max_tokens=cfg["max_tokens"],
                temperature=cfg["temperature"], top_k=cfg["top_k"],
                seed=cfg["seed"] + i))
        text = tokenizer.decode(tokens + out)
        print(text, flush=True)
        lines.append(text)

with open(cfg["out_path"], "w") as f:
    f.write(f"# {cfg['source']} d32 samples "
            f"(attn={cfg['attn_variant']}, T={cfg['temperature']}, "
            f"top_k={cfg['top_k']}, max_tokens={cfg['max_tokens']})\n")
    f.write("\n".join(lines) + "\n")
print(f"[d32-sample] wrote {len(lines)} samples", flush=True)
'''


@app.function(
    image=image,
    gpu="A10G",
    cpu=8,
    memory=32768,
    timeout=60 * 60,
    scaledown_window=5,
    volumes={CACHE_DIR: ckpt_vol},
    secrets=[hf_secret],
)
def sample_d32(
    source: str = "converted",   # "converted" (AMAP run) | "original" (Karpathy seed)
    hf_repo: str = HF_REPO_DEFAULT,
    run_tag: str = "d32ft-amap",
    step: int = -1,              # converted only: -1 = newest on Volume
    prompts: str = "",           # semicolon-separated; empty = default 7
    num_samples: int = 3,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    seed: int = 42,
) -> dict:
    """Temperature-controlled sampling for the conversion experiment's
    generation panel. 'original' loads the Karpathy seed under standard
    attention (same graft-neutral vintage handling as the trainer, so it is
    the same function the probe evaluates); 'converted' loads an AMAP
    checkpoint from the Volume. Both use the source tokenizer sandbox and
    naive cache-free generate()."""
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN", "")
    api = HfApi(token=token) if token else None
    _ensure_source_tokenizer()
    ckpt_vol.reload()

    ckpt_dir = Path(CACHE_DIR) / "base_checkpoints" / run_tag
    local_names = ([p.name for p in ckpt_dir.iterdir() if p.is_file()]
                   if ckpt_dir.exists() else [])
    steps = _checkpoint_steps(local_names, "")
    assert steps, (f"no converted checkpoint on Volume under {ckpt_dir} — "
                   "sample after at least one training run")
    tmpl_step = step if (step > 0 and step in steps) else steps[-1]
    template_meta = ckpt_dir / f"meta_{tmpl_step:06d}.json"

    if source == "original":
        model_path = _download_karpathy_seed()
        attn_variant = "standard"
        label = "original"
    else:
        model_path = ckpt_dir / f"model_{tmpl_step:06d}.pt"
        attn_variant = "hmap"
        label = f"{run_tag}-step{tmpl_step}"

    default_prompts = [
        "The capital of France is",
        "The chemical symbol of gold is",
        "If yesterday was Friday, then tomorrow will be",
        "The opposite of hot is",
        "The planets of the solar system are:",
        "My favorite color is",
        "If 5*x + 3 = 13, then x is",
    ]
    prompt_list = ([p.strip() for p in prompts.split(";") if p.strip()]
                   if prompts else default_prompts)

    workdir = Path("/tmp/d32_sample")
    workdir.mkdir(parents=True, exist_ok=True)
    out_path = workdir / f"{label}_T{temperature}_samples.txt"
    job = {
        "source": source, "attn_variant": attn_variant,
        "template_meta": str(template_meta), "model_path": str(model_path),
        "prompts": prompt_list, "num_samples": num_samples,
        "max_tokens": max_tokens, "temperature": temperature,
        "top_k": top_k, "seed": seed, "out_path": str(out_path),
    }
    (workdir / "job.json").write_text(json.dumps(job))
    (workdir / "sample_script.py").write_text(SAMPLE_PY)

    _run_streamed(
        f"cd {REPO_DIR} && PYTHONPATH={REPO_DIR} "
        f"{VENV}/bin/python -u {workdir}/sample_script.py {workdir}/job.json"
    )

    if api is not None:
        dest = f"samples/{label}_T{temperature}_samples.txt"
        api.upload_file(path_or_fileobj=str(out_path), path_in_repo=dest,
                        repo_id=hf_repo)
        print(f"[d32-sample] uploaded -> {hf_repo}/{dest}")
    print("[d32-sample] all done, returning.")
    return {"ok": True, "source": source, "label": label}


@app.local_entrypoint()
def main(
    hf_repo: str = HF_REPO_DEFAULT,
    run_tag: str = "d32ft-amap",
    resume: str = "auto",
    add_steps: int = 1000,
    save_every: int = 250,
    data_shards: int = 240,
    extra_args: str = "",
):
    train_d32ft.remote(
        hf_repo=hf_repo,
        run_tag=run_tag,
        resume=resume,
        add_steps=add_steps,
        save_every=save_every,
        data_shards=data_shards,
        extra_args=extra_args,
    )
