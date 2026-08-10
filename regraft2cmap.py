"""
regraft2cmap.py

CMAP graft / continued-pretraining harness: take the regrafted standard-
attention d32 weights from sparsetrace/amap2nanochat (the amap2nanochat run)
and continue training them under CMAP — the indefinite-kinetic + Doob corner
of the (beta, alpha) operator square:

    s(beta,alpha) = 1/2<m,m> - beta*1/2<n,n>
                    + (1-alpha)*1/2(<q,k>-<k,q>) + alpha*1/2(g_i - g_j)

    (0,0)=AMAP  (0,1)=DMAP  (1,0)=standard(eager)  (1,1)=CMAP

Lineage:  karpathy d32 (standard) -> d32ft (AMAP) -> amap2nanochat (standard)
          -> THIS RUN (CMAP: --attn-variant=hmap --hmap-alpha=1.0 --hmap-beta=1.0)

FIRST launch, if the output repo has no resume-capable checkpoint:
- install Karpathy's exact d32 tokenizer
- resolve the newest (or requested) regrafted checkpoint: Volume first
  (base_checkpoints/<seed-run-tag>/ on the shared d32ft Volume), else the
  seed HF repo (checkpoints/<seed-run-tag>/ preferred, latest/ fallback)
- load it as WEIGHTS ONLY into a depth-32 model with the CMAP operator
- create a fresh optimizer and fresh base-data stream at local step 0

LATER launches: resume this experiment's model/optimizer/loop state exactly.

The warm start is STRICT: hmap with witten=False adds no parameters at any
(beta, alpha), so the regrafted checkpoint is key-for-key identical to the
CMAP model. Any missing/unexpected/shape-mismatched key means a wrong seed
and aborts.

REQUIREMENTS on the fork (hard-checked at launch):
- nanochat/gpt.py patched with hmap_beta (apply_cmap_patch.py)
- scripts/base_train.py exposes --hmap-beta and wires it into GPTConfig,
  mirroring the existing --hmap-alpha plumbing (two lines)

CMAP is EAGER (explicit (B,H,T,T) logits): FA3/kv-cache are unavailable,
memory scales with T^2 — device-batch-size 4 at d32, and CORE eval of the
CMAP arm uses the chunked eager path (budget ~1-2h). The regrafted seed arm
evaluates fast under FA3.

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

app = modal.App("nanochat-regraft2cmap")

GPU_CONFIG = "H100"  # overridden per invocation by Function.with_options(...)

HF_REPO_DEFAULT = "sparsetrace/regraft2cmap"

# Seed: the AMAP reconditioning experiment's repo/run
SEED_REPO_DEFAULT = "sparsetrace/amap2nanochat"
SEED_RUN_TAG_DEFAULT = "amap2nanochat"

# Tokenizer still comes from Karpathy's original release: the d32ft
# embedding/unembedding rows are (transitively) trained against it.
TOKENIZER_REPO = "karpathy/nanochat-d32"
SOURCE_TOKENIZER = ("tokenizer.pkl", "token_bytes.pt")

CACHE_DIR = "/root/.cache/nanochat-d32ft"
REPO_DIR = "/root/nanochat"
VENV = f"{REPO_DIR}/.venv"

# SAME named Volume as the d32ft app, on purpose: the tokenizer, dataset
# shards, compile caches, and (crucially) the AMAP checkpoints under
# base_checkpoints/d32ft-amap/ are already there. Checkpoint dirs are keyed
# by run_tag, so this run writes to base_checkpoints/regraft2cmap/ and
# nothing collides.
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
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            # Reuse the d32ft compile caches on the shared Volume; the
            # standard-attention graph compiles fresh once, then relaunches
            # reuse it.
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
            "modal_train.py", "modal_d32ft.py", "modal_sample.py",
            "modal_eval.py", "amap2nanochat.py", "regraft2cmap.py",
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
    print(f"[r2c] running: {cmd}", flush=True)
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
    """GPU probe via the repo venv (the harness process itself has no torch)."""
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
    """Ephemeral trainer: base_train.py + one --init-from-model flag.
    STRICT warm start (see module docstring): an AMAP (coupled) checkpoint
    from this fork has exactly this model's key set; anything else aborts."""
    src = Path(REPO_DIR) / "scripts" / "base_train.py"
    dst = Path(REPO_DIR) / "scripts" / "_regraft2cmap_base_train.py"
    text = src.read_text()

    if "hmap-beta" not in text and "hmap_beta" not in text:
        raise RuntimeError(
            "base_train.py has no --hmap-beta plumbing. Add it exactly like "
            "--hmap-alpha (argparse flag + GPTConfig wiring) and re-launch; "
            "otherwise CMAP would silently train at beta=0 (= DMAP)."
        )
    gpt_text = (Path(REPO_DIR) / "nanochat" / "gpt.py").read_text()
    if "hmap_beta" not in gpt_text:
        raise RuntimeError(
            "nanochat/gpt.py has no hmap_beta — run apply_cmap_patch.py "
            "on the fork before launching."
        )

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

    warmstart = r"""
elif args.init_from_model:
    print0(f"[r2c] Weights-only warm start from {args.init_from_model}")
    print0("[r2c] New optimizer + new dataloader state; local step starts at 0")
    _init_state = torch.load(
        args.init_from_model,
        map_location=device,
        weights_only=True,
    )
    _own = model.state_dict()
    _missing = sorted(set(_own) - set(_init_state))
    _unexpected = sorted(set(_init_state) - set(_own))
    for _k in _missing:
        print0(f"[r2c]   missing from checkpoint: {_k}")
    for _k in _unexpected:
        print0(f"[r2c]   unexpected in checkpoint: {_k}")
    _shape_bad = [
        (_k, tuple(_init_state[_k].shape), tuple(_own[_k].shape))
        for _k in set(_own) & set(_init_state)
        if tuple(_init_state[_k].shape) != tuple(_own[_k].shape)
    ]
    for _k, _cs, _ms in _shape_bad:
        print0(f"[r2c]   SHAPE MISMATCH: {_k} ckpt{_cs} vs model{_ms}")
    # STRICT: the coupled-AMAP checkpoint must be key-for-key identical to
    # this standard-attention model. Missing keys => vintage/wrong-fork seed;
    # unexpected keys => witten=True run (c_qw/c_kw) or foreign file;
    # shape mismatch => wrong tokenizer or depth.
    assert not _shape_bad, \
        "[r2c] shape mismatch — wrong tokenizer or model config for this seed"
    assert not _unexpected, \
        "[r2c] seed has keys this model lacks (witten=True run?) — refusing"
    assert not _missing, \
        "[r2c] seed lacks keys this model has (vintage checkpoint?) — refusing"
    model.load_state_dict(_init_state, strict=True)
    print0("[r2c] strict load OK: AMAP weights grafted under standard attention")
    del _init_state, _own
"""
    text = text.replace(load_anchor, load_anchor + warmstart, 1)
    dst.write_text(text)
    print(f"[r2c] installed ephemeral trainer: {dst}")
    return "scripts._regraft2cmap_base_train"


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
        _hf_download_into(TOKENIZER_REPO, name, tok_dir / name, token=None)
    print(f"[r2c] installed exact Karpathy d32 tokenizer -> {tok_dir}")


def _checkpoint_steps(files: list[str], prefix: str) -> list[int]:
    """Steps with model_ AND meta_ AND optim_ (resume-capable)."""
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
        m = re.fullmatch(r"optim_(\d+)(?:_rank\d+)?\.pt", name)
        if m:
            optim_steps.add(int(m.group(1)))
    return sorted(model_steps & meta_steps & optim_steps)


def _weights_steps(files: list[str], prefix: str) -> list[int]:
    """Steps with model_ AND meta_ (optimizer not required — the seed is a
    weights-only warm start, so the d32ft optimizer state is irrelevant)."""
    model_steps, meta_steps = set(), set()
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
    return sorted(model_steps & meta_steps)


def _resolve_regraft_seed(
    api, token: str, seed_repo: str, seed_run_tag: str, seed_step: int
):
    """Locate the regrafted seed checkpoint (model_*.pt + meta_*.json).

    Volume first: the shared Volume normally still holds the amap2nanochat run's
    checkpoints under base_checkpoints/<seed_run_tag>/ — reading them costs
    seconds vs ~20GB of HF download at GPU rates. HF is the fallback:
    checkpoints/<seed_run_tag>/ preferred over latest/ (latest/ can be
    overwritten by whichever run pushed last on the seed repo).

    Returns (model_path, meta_path, step, origin_str).
    """
    seed_dir = Path(CACHE_DIR) / "base_checkpoints" / seed_run_tag
    local_names = ([p.name for p in seed_dir.iterdir() if p.is_file()]
                   if seed_dir.exists() else [])
    local_steps = _weights_steps(local_names, "")
    if local_steps and (seed_step in local_steps or seed_step < 0):
        step = seed_step if seed_step > 0 else local_steps[-1]
        model_path = seed_dir / f"model_{step:06d}.pt"
        meta_path = seed_dir / f"meta_{step:06d}.json"
        origin = f"volume:{seed_run_tag}:step-{step}"
        print(f"[r2c] seed (volume, no download): {model_path}")
        return model_path, meta_path, step, origin

    files = list(api.list_repo_files(repo_id=seed_repo))
    candidates = [
        (f"checkpoints/{seed_run_tag}/",
         _weights_steps(files, f"checkpoints/{seed_run_tag}/")),
        ("latest/", _weights_steps(files, "latest/")),
    ]
    chosen_prefix, step = None, -1
    for prefix, steps in candidates:
        if not steps:
            continue
        if seed_step > 0 and seed_step in steps:
            chosen_prefix, step = prefix, seed_step
            break
        if seed_step < 0 and steps[-1] > step:
            chosen_prefix, step = prefix, steps[-1]
    if chosen_prefix is None:
        raise RuntimeError(
            f"No regrafted seed checkpoint found: neither the Volume ({seed_dir}) "
            f"nor {seed_repo} has "
            f"{'step ' + str(seed_step) if seed_step > 0 else 'any step'} "
            f"under checkpoints/{seed_run_tag}/ or latest/"
        )

    dest_dir = Path(CACHE_DIR) / "bootstrap" / "regraft-seed" / f"{step:06d}"
    model_path = _hf_download_into(
        seed_repo, f"{chosen_prefix}model_{step:06d}.pt",
        dest_dir / f"model_{step:06d}.pt", token=token,
    )
    meta_path = _hf_download_into(
        seed_repo, f"{chosen_prefix}meta_{step:06d}.json",
        dest_dir / f"meta_{step:06d}.json", token=token,
    )
    origin = f"{seed_repo}:{chosen_prefix}step-{step}"
    print(f"[r2c] seed (HF download): {origin} -> {model_path}")
    return model_path, meta_path, step, origin


def _validate_seed_meta(meta_path: Path, seed_run_tag: str):
    """The seed must be the regrafted standard-attention run: depth 32,
    attn standard. Guards against grabbing a foreign checkpoint out of
    latest/. Metas that predate some fields are tolerated (None passes)."""
    meta = json.loads(meta_path.read_text())
    uc = meta.get("user_config", {}) or {}
    if int(uc.get("depth", -1)) != 32:
        raise RuntimeError(f"Refusing non-d32 seed: depth={uc.get('depth')}")
    if uc.get("model_tag") not in (None, seed_run_tag):
        raise RuntimeError(
            f"Refusing seed model_tag={uc.get('model_tag')!r}; "
            f"expected {seed_run_tag!r}"
        )
    variant = uc.get("attn_variant")
    if variant not in (None, "standard"):
        raise RuntimeError(
            f"Seed attn_variant={variant!r} is not the regrafted standard run"
        )
    print("[r2c] seed meta validated: depth=32, attn=standard (regrafted)")


def _restore_own_checkpoint(api, hf_repo: str, token: str, run_name: str):
    from huggingface_hub import hf_hub_download

    ckpt_dir = Path(CACHE_DIR) / "base_checkpoints" / run_name

    # Volume first (the previous run saved here); HF is the durable mirror.
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
        use_step = best_local
        print(
            f"[r2c] RESUME (volume, no download): step {use_step} "
            f"already in {ckpt_dir}"
        )
    else:
        use_step = chosen_step
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        step_token = f"{use_step:06d}"
        print(
            f"[r2c] RESUME (HF download): {hf_repo}/{chosen_prefix} "
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

    print(f"[r2c] RESUME: step {use_step} -> {ckpt_dir}")
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
            print("[r2c] volume committed")
        except Exception as e:
            print(f"[r2c] volume commit failed (non-fatal): {e}")

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
                print("[r2c] mirror cycle skipped (training finished)")
                break

            if ops:
                api.create_commit(
                    repo_id=hf_repo,
                    operations=ops,
                    commit_message=f"r2c mirror latest ({len(ops)} files)",
                )
                for name, mtime in names:
                    pushed_mtimes[name] = mtime
                print(f"[r2c] mirrored {len(ops)} file(s) -> latest/")
        except Exception as e:
            print(f"[r2c] HF mirror failed (non-fatal): {e}")


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
    """Detached CPU stage-out (same rationale as d32ft: never idle 8xH100
    through a ~30GB upload)."""
    from huggingface_hub import HfApi, CommitOperationAdd

    token = os.environ.get("HF_TOKEN", "")
    api = HfApi(token=token)
    ckpt_vol.reload()

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
            commit_message=f"{run_tag}: CMAP graft through step {horizon}",
        )
        print(f"[r2c-upload] pushed {len(ops)} file(s) in one commit")

    # Prune superseded steps from THIS run's dir only. The seed run's
    # checkpoints (base_checkpoints/<seed_run_tag>/) are never touched.
    if ckpt_dir.exists():
        step_re = re.compile(r"(?:model|meta|optim)_(\d+)")
        steps = sorted({int(m.group(1)) for p in ckpt_dir.iterdir()
                        if (m := step_re.match(p.name))})
        for old in steps[:-keep_last_steps] if keep_last_steps > 0 else []:
            tok = f"{old:06d}"
            for p in list(ckpt_dir.iterdir()):
                if tok in p.name:
                    p.unlink()
            print(f"[r2c-upload] pruned superseded step {old} from Volume")
        ckpt_vol.commit()

    print("[r2c-upload] all done, returning.")
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
def train_regraft2cmap(
    hf_repo: str = HF_REPO_DEFAULT,
    run_tag: str = "regraft2cmap",
    num_gpus: int = 1,
    resume: str = "auto",
    seed_repo: str = SEED_REPO_DEFAULT,
    seed_run_tag: str = SEED_RUN_TAG_DEFAULT,
    seed_step: int = -1,
    add_steps: int = 1000,
    save_every: int = 250,
    data_shards: int = 240,
    extra_args: str = "",
) -> dict:
    if add_steps <= 0:
        raise ValueError("regraft2cmap requires add_steps > 0")
    if resume not in {"auto", "never", "force"}:
        raise ValueError("resume must be auto|never|force")
    if num_gpus not in {1, 2, 4, 8}:
        raise ValueError("num_gpus must be one of 1, 2, 4, 8")

    os.chdir(REPO_DIR)

    visible_gpus = _count_visible_gpus()
    print(
        f"[r2c] GPU sanity: requested={num_gpus}, "
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
        raise RuntimeError(
            "HF_TOKEN is required (private seed repo + private output repo)"
        )

    from huggingface_hub import HfApi, CommitOperationAdd

    api = HfApi(token=token)
    api.create_repo(repo_id=hf_repo, private=True, exist_ok=True)

    print("[r2c] stage: installing exact Karpathy d32 tokenizer...", flush=True)
    _ensure_source_tokenizer()
    print("[r2c] stage complete: tokenizer ready", flush=True)

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
    seed_used_step = -1

    restored = None
    if resume != "never":
        print(
            f"[r2c] stage: checking {hf_repo} for a resume-capable checkpoint...",
            flush=True,
        )
        restored = _restore_own_checkpoint(api, hf_repo, token, run_tag)
        if restored is None:
            print("[r2c] stage complete: no own checkpoint found", flush=True)
        else:
            print("[r2c] stage complete: own checkpoint found", flush=True)
    else:
        print("[r2c] resume='never' — skipping own-checkpoint lookup", flush=True)

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
            f"[r2c] stage: resolving regrafted seed "
            f"({seed_repo} / {seed_run_tag} / step {seed_step})...",
            flush=True,
        )
        seed_model, seed_meta, seed_used_step, origin = _resolve_regraft_seed(
            api, token, seed_repo, seed_run_tag, seed_step
        )
        _validate_seed_meta(seed_meta, seed_run_tag)
        print("[r2c] stage complete: regrafted seed ready", flush=True)
        init_flag = f"--init-from-model={seed_model} "
        print(
            "[r2c] BOOTSTRAP: regrafted standard-attention d32 weights -> CMAP "
            "operator (eager); optimizer/dataloader start fresh at step 0"
        )

    python = f"{VENV}/bin/python"
    torchrun = f"{VENV}/bin/torchrun"

    print(
        f"[r2c] stage: ensuring {data_shards} nanochat dataset shards are available...",
        flush=True,
    )
    _run_streamed(f"{python} -m nanochat.dataset -n {data_shards}")
    print("[r2c] stage complete: dataset ready", flush=True)

    print(
        "[r2c] stage: installing ephemeral weights-only warmstart trainer...",
        flush=True,
    )
    trainer_module = _install_warmstart_train_module()
    print(f"[r2c] stage complete: trainer module = {trainer_module}", flush=True)

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
    # NOTE: --attn-variant / --hmap-alpha / --hmap-beta / --window-pattern
    # arrive via extra_args from the workflow (VARIANT_ARGS) so the recipe
    # lives in one place; only depth, identity, cadence and horizon are
    # pinned here. The workflow sets hmap alpha=1 beta=1 (CMAP), window L,
    # and device-batch-size=4 (eager T^2 logits; same footprint as the
    # validated AMAP d32 runs).
    cmd = (
        f"{launcher} "
        f"--depth=32 "
        f"--model-tag={run_tag} "
        f"--save-every={save_every} "
        f"--num-iterations={horizon} "
        f"{resume_flag}"
        f"{init_flag}"
        f"{extra_args}"
    )

    print(f"[r2c] origin: {origin}", flush=True)
    print(
        f"[r2c] additive budget: {add_steps} new steps "
        f"({resumed_step} -> {horizon})",
        flush=True,
    )
    print(
        "[r2c] stage: launching depth-32 CMAP (eager) training process "
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
            print("[r2c] waiting for background mirror cycle to finish...")
            sync_thread.join(timeout=600)
        trained_rows = parse_log_to_csv(log_path, csv_path)
        n_icl = parse_icl_to_csv(log_path, icl_csv_path)
        ns = extract_samples(log_path, samples_path, run_tag)
        print(f"[r2c] parsed {trained_rows} step lines; {n_icl} ICL lines; "
              f"{ns} samples")
        ckpt_vol.commit()

    # Zero-step guard (resume with nothing left to do: don't re-push ~20GB).
    if trained_rows == 0 and resume_flag:
        print("[r2c] 0 training steps on a resume — skipping final upload.")
        print("[r2c] all done, returning. (Anything after this line is "
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
                "seed_repo": seed_repo,
                "seed_run_tag": seed_run_tag,
                "seed_step": seed_used_step,
                "attention": "hmap alpha=1.0 beta=1.0 (CMAP, eager)",
                "graft": "regrafted standard -> CMAP (hmap alpha=1 beta=1); "
                         "strict key-for-key load, no neutralization needed",
                "depth": 32,
                "resumed_step": resumed_step,
                "horizon": horizon,
                "new_steps_requested": add_steps,
                "num_gpus": nproc,
                "data_shards_requested": data_shards,
                "extra_args": extra_args,
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
            commit_message=f"{run_tag}: CMAP graft through step {horizon}",
        )

    elapsed_h = (time.time() - t0) / 3600
    print(f"[r2c] finished in {elapsed_h:.2f} h")
    print("[r2c] all done, returning. (Anything after this line is "
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


# ── CORE evaluation: AMAP seed AND regrafted standard-attention checkpoints ─
CORE_EVAL_PY = r'''
import json, sys, torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import compute_init

cfg = json.load(open(sys.argv[1]))
template = json.load(open(cfg["template_meta"]))["model_config"]
template.update(cfg["attn_overrides"])
print(f"[r2c-core] source={cfg['source']} attn_variant={cfg['attn_variant']}",
      flush=True)

ddp, rank, local_rank, world, device = compute_init("cuda")

model = GPT(GPTConfig(**template)).to(device)
state = torch.load(cfg["model_path"], map_location=device, weights_only=True)
own = model.state_dict()
missing = sorted(set(own) - set(state))
unexpected = sorted(set(state) - set(own))
assert not missing, f"missing keys: {missing[:5]}"
assert not unexpected, f"unexpected keys: {unexpected[:5]}"
model.load_state_dict(state, strict=True)
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
print(f"[r2c-core] CORE metric: {core['core_metric']:.4f}", flush=True)
with open(cfg["out_path"], "w") as f:
    json.dump(result, f, indent=2)
print(f"[r2c-core] wrote {cfg['out_path']}", flush=True)
'''


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=4 * 60 * 60,
    scaledown_window=5,
    volumes={CACHE_DIR: ckpt_vol},
    secrets=[hf_secret],
)
def core_eval_r2c(
    source: str = "cmap",   # "cmap" (CMAP, eager) | "seed" (regrafted, FA3)
    hf_repo: str = HF_REPO_DEFAULT,
    run_tag: str = "regraft2cmap",
    seed_repo: str = SEED_REPO_DEFAULT,
    seed_run_tag: str = SEED_RUN_TAG_DEFAULT,
    step: int = -1,              # -1 = newest available (for either source)
    max_per_task: int = 500,
) -> dict:
    """CORE metric on a fresh GPU. 'regrafted' = this run's checkpoint under
    standard attention (FA3, fast); 'seed' = the AMAP checkpoint under the
    chunked eager hmap path (slow: budget ~1-2h). Both are strict loads of
    full-key fork checkpoints — no vintage handling needed on this side."""
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN", "")
    api = HfApi(token=token) if token else None
    _ensure_source_tokenizer()
    ckpt_vol.reload()

    if source == "seed":
        model_path, template_meta, use_step, _ = _resolve_regraft_seed(
            api, token, seed_repo, seed_run_tag, step
        )
        attn_variant = "standard"
        label = f"{seed_run_tag}-step{use_step}-seed"
    else:
        ckpt_dir = Path(CACHE_DIR) / "base_checkpoints" / run_tag
        local_names = ([p.name for p in ckpt_dir.iterdir() if p.is_file()]
                       if ckpt_dir.exists() else [])
        steps = _weights_steps(local_names, "")
        assert steps, f"no regrafted checkpoint on Volume under {ckpt_dir}"
        use_step = step if (step > 0 and step in steps) else steps[-1]
        model_path = ckpt_dir / f"model_{use_step:06d}.pt"
        template_meta = ckpt_dir / f"meta_{use_step:06d}.json"
        attn_variant = "hmap"
        label = f"{run_tag}-step{use_step}"

    workdir = Path("/tmp/r2c_core")
    workdir.mkdir(parents=True, exist_ok=True)
    out_path = workdir / f"{label}_core.json"
    attn_overrides = (
        {"attn_variant": "hmap", "hmap_alpha": 1.0, "hmap_beta": 1.0}
        if attn_variant == "hmap" else {"attn_variant": "standard"}
    )
    job = {
        "source": source, "attn_variant": attn_variant,
        "attn_overrides": attn_overrides,
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
        print(f"[r2c-core] uploaded -> {hf_repo}/{dest}")
    print("[r2c-core] all done, returning.")
    return json.loads(out_path.read_text())


# ── Sampling: AMAP seed AND regrafted standard-attention checkpoints ────────
SAMPLE_PY = r'''
import json, sys, torch
from pathlib import Path
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer

cfg = json.load(open(sys.argv[1]))
template = json.load(open(cfg["template_meta"]))["model_config"]
template.update(cfg["attn_overrides"])
print(f"[r2c-sample] source={cfg['source']} attn_variant={cfg['attn_variant']}",
      flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT(GPTConfig(**template)).to(device)
state = torch.load(cfg["model_path"], map_location=device, weights_only=True)
own = model.state_dict()
missing = sorted(set(own) - set(state))
unexpected = sorted(set(state) - set(own))
assert not missing, f"missing keys: {missing[:5]}"
assert not unexpected, f"unexpected keys: {unexpected[:5]}"
model.load_state_dict(state, strict=True)
model.eval()

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
print(f"[r2c-sample] wrote {len(lines)} samples", flush=True)
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
def sample_r2c(
    source: str = "cmap",   # "cmap" (CMAP, eager) | "seed" (regrafted, FA3)
    hf_repo: str = HF_REPO_DEFAULT,
    run_tag: str = "regraft2cmap",
    seed_repo: str = SEED_REPO_DEFAULT,
    seed_run_tag: str = SEED_RUN_TAG_DEFAULT,
    step: int = -1,
    prompts: str = "",           # semicolon-separated; empty = default 7
    num_samples: int = 3,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    seed: int = 42,
) -> dict:
    """Temperature-controlled sampling for the regraft's generation panel,
    with the regrafted seed available as the comparison arm. Both use the source
    tokenizer sandbox and naive cache-free generate()."""
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN", "")
    api = HfApi(token=token) if token else None
    _ensure_source_tokenizer()
    ckpt_vol.reload()

    if source == "seed":
        model_path, template_meta, use_step, _ = _resolve_regraft_seed(
            api, token, seed_repo, seed_run_tag, step
        )
        attn_variant = "standard"
        label = f"{seed_run_tag}-step{use_step}-seed"
    else:
        ckpt_dir = Path(CACHE_DIR) / "base_checkpoints" / run_tag
        local_names = ([p.name for p in ckpt_dir.iterdir() if p.is_file()]
                       if ckpt_dir.exists() else [])
        steps = _weights_steps(local_names, "")
        assert steps, (f"no regrafted checkpoint on Volume under {ckpt_dir} — "
                       "sample after at least one training run")
        use_step = step if (step > 0 and step in steps) else steps[-1]
        model_path = ckpt_dir / f"model_{use_step:06d}.pt"
        template_meta = ckpt_dir / f"meta_{use_step:06d}.json"
        attn_variant = "hmap"
        label = f"{run_tag}-step{use_step}"

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

    workdir = Path("/tmp/r2c_sample")
    workdir.mkdir(parents=True, exist_ok=True)
    out_path = workdir / f"{label}_T{temperature}_samples.txt"
    attn_overrides = (
        {"attn_variant": "hmap", "hmap_alpha": 1.0, "hmap_beta": 1.0}
        if attn_variant == "hmap" else {"attn_variant": "standard"}
    )
    job = {
        "source": source, "attn_variant": attn_variant,
        "attn_overrides": attn_overrides,
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
        print(f"[r2c-sample] uploaded -> {hf_repo}/{dest}")
    print("[r2c-sample] all done, returning.")
    return {"ok": True, "source": source, "label": label}


@app.local_entrypoint()
def main(
    hf_repo: str = HF_REPO_DEFAULT,
    run_tag: str = "regraft2cmap",
    resume: str = "auto",
    seed_repo: str = SEED_REPO_DEFAULT,
    seed_run_tag: str = SEED_RUN_TAG_DEFAULT,
    seed_step: int = -1,
    add_steps: int = 1000,
    save_every: int = 250,
    data_shards: int = 240,
    extra_args: str = "",
):
    train_regraft2cmap.remote(
        hf_repo=hf_repo,
        run_tag=run_tag,
        resume=resume,
        seed_repo=seed_repo,
        seed_run_tag=seed_run_tag,
        seed_step=seed_step,
        add_steps=add_steps,
        save_every=save_every,
        data_shards=data_shards,
        extra_args=extra_args,
    )
