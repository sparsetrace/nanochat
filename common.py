from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
import modal

CACHE_DIR = "/root/.cache/nanochat-d32ft"
REPO_DIR = "/root/nanochat"
VENV = f"{REPO_DIR}/.venv"

TOKENIZER_REPO = "karpathy/nanochat-d32"
SOURCE_TOKENIZER = ("tokenizer.pkl", "token_bytes.pt")

RESULTS_REPO = "sparsetrace/DAC-d32-results"

STATE_ALIASES = {
    "original": dict(repo="karpathy/nanochat-d32", run_tag=None, operator="standard",
                     fixed_model="model_000650.pt", fixed_meta="meta_000650.json"),
    "amap1": dict(repo="sparsetrace/d32ft", run_tag="d32ft-amap", operator="hmap"),
    "attention1": dict(repo="sparsetrace/amap2nanochat", run_tag="amap2nanochat", operator="standard"),
    "amap2": dict(repo="sparsetrace/nanochat2amap", run_tag="nanochat2amap", operator="hmap"),
}

ckpt_vol = modal.Volume.from_name("nanochat-d32ft-cache", create_if_missing=True, version=2)
hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "build-essential")
    .pip_install("huggingface_hub")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
    .env({
        "PATH": "/root/.cargo/bin:/root/.local/bin:/usr/local/bin:/usr/bin:/bin",
        "OMP_NUM_THREADS": "1",
        "HF_HUB_DISABLE_XET": "1",
        "NANOCHAT_BASE_DIR": CACHE_DIR,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCHINDUCTOR_CACHE_DIR": f"{CACHE_DIR}/inductor_cache",
        "TRITON_CACHE_DIR": f"{CACHE_DIR}/triton_cache",
    })
    .add_local_dir(
        ".",
        remote_path=REPO_DIR,
        copy=True,
        ignore=[
            ".git", ".venv", "__pycache__", "*.pyc", ".github",
            "modal_train.py", "modal_d32ft.py", "modal_sample.py", "modal_eval.py",
            "amap2nanochat.py", "nanochat2amap.py", "icl_probe.py",
            "diagnostics.py", "common.py",
        ],
    )
    .run_commands(f"cd {REPO_DIR} && uv sync --extra gpu")
)

def hf_download_into(repo_id: str, filename: str, dest: Path, token=None) -> Path:
    from huggingface_hub import hf_hub_download
    cached = hf_hub_download(
        repo_id=repo_id, filename=filename, token=token,
        local_dir=str(Path(CACHE_DIR) / "hf_downloads" / repo_id.replace("/", "__")),
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, dest)
    return dest

def ensure_source_tokenizer() -> None:
    tok_dir = Path(CACHE_DIR) / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_TOKENIZER:
        hf_download_into(TOKENIZER_REPO, name, tok_dir / name, token=None)
    print(f"[diag] tokenizer ready -> {tok_dir}", flush=True)

def _weights_steps(files: list[str], prefix: str) -> list[int]:
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

def resolve_state(api, token: str, state: str, repo: str="", run_tag: str="", step: int=-1):
    preset = dict(STATE_ALIASES.get(state, {}))
    if repo:
        preset["repo"] = repo
    if run_tag:
        preset["run_tag"] = run_tag
    if "repo" not in preset:
        raise ValueError(f"Unknown state {state!r}; provide repo/run_tag explicitly.")

    use_repo = preset["repo"]
    use_run = preset.get("run_tag")

    if use_run is None:
        dest = Path(CACHE_DIR) / "diagnostics" / state
        model = hf_download_into(use_repo, preset["fixed_model"], dest / preset["fixed_model"], token=None)
        meta = hf_download_into(use_repo, preset["fixed_meta"], dest / preset["fixed_meta"], token=None)
        return dict(state=state, repo=use_repo, run_tag=None, step=650,
                    operator_hint=preset.get("operator"), model_path=model, meta_path=meta,
                    origin=f"{use_repo}:{preset['fixed_model']}")

    ckpt_dir = Path(CACHE_DIR) / "base_checkpoints" / use_run
    local = [p.name for p in ckpt_dir.iterdir() if p.is_file()] if ckpt_dir.exists() else []
    steps = _weights_steps(local, "")
    if steps and (step < 0 or step in steps):
        use = steps[-1] if step < 0 else step
        return dict(state=state, repo=use_repo, run_tag=use_run, step=use,
                    operator_hint=preset.get("operator"),
                    model_path=ckpt_dir / f"model_{use:06d}.pt",
                    meta_path=ckpt_dir / f"meta_{use:06d}.json",
                    origin=f"volume:{use_run}:step-{use}")

    files = list(api.list_repo_files(repo_id=use_repo))
    for prefix in (f"checkpoints/{use_run}/", "latest/"):
        steps = _weights_steps(files, prefix)
        if not steps:
            continue
        use = steps[-1] if step < 0 else step
        if use not in steps:
            continue
        dest = Path(CACHE_DIR) / "diagnostics" / use_run / f"{use:06d}"
        model = hf_download_into(use_repo, f"{prefix}model_{use:06d}.pt",
                                 dest / f"model_{use:06d}.pt", token=token)
        meta = hf_download_into(use_repo, f"{prefix}meta_{use:06d}.json",
                                dest / f"meta_{use:06d}.json", token=token)
        return dict(state=state, repo=use_repo, run_tag=use_run, step=use,
                    operator_hint=preset.get("operator"), model_path=model, meta_path=meta,
                    origin=f"{use_repo}:{prefix}step-{use}")
    raise RuntimeError(f"Checkpoint not found for state={state} repo={use_repo} run_tag={use_run} step={step}")

def upload_json(api, path: Path, label: str, kind: str, results_repo: str) -> str:
    api.create_repo(repo_id=results_repo, private=True, exist_ok=True)
    dest = f"{kind}/{label}.json"
    api.upload_file(path_or_fileobj=str(path), path_in_repo=dest, repo_id=results_repo)
    print(f"[diag] uploaded -> {results_repo}/{dest}", flush=True)
    return dest
