"""
icl_probe.py — standalone ICL/induction probe on a PINNED 240-shard eval set.

    modal run icl_probe.py --arm amap                  # annealed d32ft-amap, newest step
    modal run icl_probe.py --arm original              # Karpathy seed, standard attn
    modal run icl_probe.py --arm cmap --step 3000
    modal run icl_probe.py --arm regrafted

WHY THIS EXISTS: the d32ft finishing anneal ran with data_shards=60 while every
other launch used 240, so its ICL early/late rows (steps 1600-3000) were
measured on a different eval stream and are NOT comparable to the other arms
(the seeded synthetic induction probe was unaffected — which is how the
mismatch was caught: identical induction triplets, contradictory ICL scores at
step 1600). This harness re-probes any checkpoint on the 240-shard set so all
endpoints sit on ONE probe.

MECHANISM: no new eval code. The checked-in scripts/base_train.py already runs
the val-bpb and ICL/induction probes at the first loop step. We install the
same ephemeral --init-from-model trainer as the training harnesses (vintage
strict-with-allowlist + graft-neutral fill, which also accepts full fork
checkpoints since missing=[] passes), then run ONE step with every LR set to
0.0: the step-0 evals fire on the loaded weights exactly, and the single
zero-LR update is a no-op. Nothing is checkpointed (save-every is huge and the
probe's scratch dir is deleted afterwards); only a small JSON goes to HF.

ARMS (checkpoint + operator presets; all overridable):
  original  : karpathy/nanochat-d32 model_000650, standard
  amap      : sparsetrace/d32ft         / d32ft-amap    , hmap a=0
  cmap      : sparsetrace/d32cmap       / d32cmap       , hmap a=1 b=1
  regrafted : sparsetrace/amap2nanochat / amap2nanochat , standard

Volume-first checkpoint resolution on the shared d32ft Volume (weights+meta
only — optimizer files are irrelevant to a weights-only probe), HF fallback.

IMPORTANT harness constraint: this function runs under the container's SYSTEM
python (modal + huggingface_hub only). torch exists ONLY inside the repo venv.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import modal

app = modal.App("nanochat-icl-probe")

PROBE_DATA_SHARDS = 240   # THE pinned probe set. Do not change casually.

TOKENIZER_REPO = "karpathy/nanochat-d32"
SOURCE_TOKENIZER = ("tokenizer.pkl", "token_bytes.pt")
KARPATHY_MODEL = "model_000650.pt"
KARPATHY_META = "meta_000650.json"

ARMS = {
    "original":  dict(repo="karpathy/nanochat-d32", run_tag=None,
                      attn_variant="standard", hmap_alpha=0.0, hmap_beta=0.0),
    "amap":      dict(repo="sparsetrace/d32ft", run_tag="d32ft-amap",
                      attn_variant="hmap", hmap_alpha=0.0, hmap_beta=0.0),
    "cmap":      dict(repo="sparsetrace/d32cmap", run_tag="d32cmap",
                      attn_variant="hmap", hmap_alpha=1.0, hmap_beta=1.0),
    "regrafted": dict(repo="sparsetrace/amap2nanochat", run_tag="amap2nanochat",
                      attn_variant="standard", hmap_alpha=0.0, hmap_beta=0.0),
}

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
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
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
            "icl_probe.py",
        ],
    )
    .run_commands(f"cd {REPO_DIR} && uv sync --extra gpu")
)

ICL_RE = re.compile(
    r"Step (\d+) \| ICL early: ([\d.]+) late: ([\d.]+) score: ([+\-][\d.]+) \| "
    r"Induction loss: ([\d.]+) acc: ([\d.]+) \(random-half: ([\d.]+)\)"
)
BPB_RE = re.compile(r"Step (\d+) \| Validation bpb: ([\d.]+)")


def _run_streamed(cmd: str, log_path: Path | None = None):
    print(f"[probe] running: {cmd}", flush=True)
    log_f = open(log_path, "a") if log_path else None
    try:
        proc = subprocess.Popen(
            cmd, shell=True, executable="/bin/bash", cwd=REPO_DIR,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
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
    probe = subprocess.run(
        [f"{VENV}/bin/python", "-c",
         "import torch; print(torch.cuda.device_count())"],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        raise RuntimeError(f"GPU probe failed: {probe.stderr.strip()}")
    return int(probe.stdout.strip())


def _hf_download_into(repo_id: str, filename: str, dest: Path, token=None):
    from huggingface_hub import hf_hub_download
    cached = hf_hub_download(
        repo_id=repo_id, filename=filename, token=token,
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
    print(f"[probe] installed exact Karpathy d32 tokenizer -> {tok_dir}")


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


def _resolve_checkpoint(api, token, repo: str, run_tag: str | None, step: int):
    """Weights+meta for the requested arm. Volume-first, HF fallback.
    Karpathy original (run_tag=None) is a fixed public file."""
    if run_tag is None:
        seed_dir = Path(CACHE_DIR) / "bootstrap" / "karpathy-d32"
        model = _hf_download_into(repo, KARPATHY_MODEL,
                                  seed_dir / KARPATHY_MODEL, token=None)
        _hf_download_into(repo, KARPATHY_META, seed_dir / KARPATHY_META, token=None)
        return model, 650, f"{repo}:{KARPATHY_MODEL}"

    ckpt_dir = Path(CACHE_DIR) / "base_checkpoints" / run_tag
    local_names = ([p.name for p in ckpt_dir.iterdir() if p.is_file()]
                   if ckpt_dir.exists() else [])
    local_steps = _weights_steps(local_names, "")
    if local_steps and (step in local_steps or step < 0):
        use = step if step > 0 else local_steps[-1]
        print(f"[probe] checkpoint (volume): {run_tag} step {use}")
        return ckpt_dir / f"model_{use:06d}.pt", use, f"volume:{run_tag}:step-{use}"

    files = list(api.list_repo_files(repo_id=repo))
    for prefix in (f"checkpoints/{run_tag}/", "latest/"):
        steps = _weights_steps(files, prefix)
        if not steps:
            continue
        use = step if (step > 0 and step in steps) else (steps[-1] if step < 0 else None)
        if use is None:
            continue
        dest = Path(CACHE_DIR) / "probe_seeds" / run_tag / f"{use:06d}"
        model = _hf_download_into(repo, f"{prefix}model_{use:06d}.pt",
                                  dest / f"model_{use:06d}.pt", token=token)
        print(f"[probe] checkpoint (HF): {repo}/{prefix} step {use}")
        return model, use, f"{repo}:{prefix}step-{use}"
    raise RuntimeError(
        f"No checkpoint with model+meta found for {run_tag} "
        f"(step {step}) on Volume or in {repo}"
    )


def _install_probe_trainer(need_beta: bool) -> str:
    """base_train.py + --init-from-model, vintage strict-with-allowlist +
    graft-neutral fill (identical policy to the training harnesses; a full
    fork checkpoint passes trivially with missing=[])."""
    src = Path(REPO_DIR) / "scripts" / "base_train.py"
    dst = Path(REPO_DIR) / "scripts" / "_icl_probe_base_train.py"
    text = src.read_text()

    if need_beta and "hmap-beta" not in text and "hmap_beta" not in text:
        raise RuntimeError(
            "base_train.py has no --hmap-beta plumbing but the requested arm "
            "needs beta>0. Add it (mirroring --hmap-alpha) or probe another arm."
        )

    arg_anchor = (
        'parser.add_argument("--resume-from-step", type=int, default=-1, '
        'help="resume training from this step (-1 = disable)")'
    )
    if arg_anchor not in text:
        raise RuntimeError("Could not patch base_train.py: argparse anchor changed.")
    text = text.replace(
        arg_anchor,
        arg_anchor
        + '\nparser.add_argument("--init-from-model", type=str, default="", '
          'help="weights-only warm start from a raw nanochat state_dict .pt")',
        1,
    )

    load_anchor = "    del model_data # free up this memory after the copy\n"
    if load_anchor not in text:
        raise RuntimeError("Could not patch base_train.py: load anchor changed.")

    warmstart = r"""
elif args.init_from_model:
    print0(f"[probe] Weights-only load from {args.init_from_model}")
    _init_state = torch.load(
        args.init_from_model, map_location=device, weights_only=True,
    )
    _own = model.state_dict()
    _missing = sorted(set(_own) - set(_init_state))
    _unexpected = sorted(set(_init_state) - set(_own))
    _shape_bad = [
        _k for _k in set(_own) & set(_init_state)
        if tuple(_init_state[_k].shape) != tuple(_own[_k].shape)
    ]
    assert not _shape_bad, f"[probe] shape mismatch: {_shape_bad[:5]}"
    assert not _unexpected, f"[probe] unexpected keys: {_unexpected[:5]}"
    _ALLOW = ("value_embed", "ve_gate", "resid_lambda", "x0_lambda",
              "smear_gate", "smear_lambda", "backout_lambda")
    _bad = [_k for _k in _missing if not any(_a in _k for _a in _ALLOW)]
    assert not _bad, f"[probe] missing keys outside vintage allowlist: {_bad[:5]}"
    model.load_state_dict(_init_state, strict=False)
    with torch.no_grad():
        for _k, _p in model.named_parameters():
            if _k not in _missing:
                continue
            if "value_embeds" in _k:
                _p.zero_()
            elif "resid_lambda" in _k:
                _p.fill_(1.0)
            elif "x0_lambda" in _k or "backout_lambda" in _k:
                _p.zero_()
    if _missing:
        print0(f"[probe] graft-neutralized {len(_missing)} vintage-absent params")
    del _init_state, _own
"""
    text = text.replace(load_anchor, load_anchor + warmstart, 1)
    dst.write_text(text)
    print(f"[probe] installed ephemeral probe trainer: {dst}")
    return "scripts._icl_probe_base_train"


@app.function(
    image=image,
    gpu="H100",
    cpu=32,
    memory=131072,
    timeout=6 * 60 * 60,
    retries=0,
    scaledown_window=5,
    volumes={CACHE_DIR: ckpt_vol},
    secrets=[hf_secret],
)
def run_probe(
    arm: str = "amap",
    step: int = -1,                 # -1 = newest available for the arm
    num_gpus: int = 1,
    repo: str = "",                 # override arm preset
    run_tag: str = "",              # override arm preset
    attn_variant: str = "",         # override arm preset
    hmap_alpha: float = -1.0,       # override arm preset (-1 = preset)
    hmap_beta: float = -1.0,        # override arm preset (-1 = preset)
    push_repo: str = "sparsetrace/d32ft",   # where the probe JSON is archived
    extra_args: str = "",
) -> dict:
    assert arm in ARMS or (repo and attn_variant), \
        f"arm must be one of {sorted(ARMS)} or fully specified via overrides"
    preset = dict(ARMS.get(arm, dict(repo="", run_tag=None, attn_variant="standard",
                                     hmap_alpha=0.0, hmap_beta=0.0)))
    if repo:
        preset["repo"] = repo
    if run_tag:
        preset["run_tag"] = run_tag
    if attn_variant:
        preset["attn_variant"] = attn_variant
    if hmap_alpha >= 0.0:
        preset["hmap_alpha"] = hmap_alpha
    if hmap_beta >= 0.0:
        preset["hmap_beta"] = hmap_beta

    os.chdir(REPO_DIR)
    visible = _count_visible_gpus()
    if visible != num_gpus:
        raise RuntimeError(f"GPU mismatch: requested {num_gpus}, torch sees {visible}")
    token = os.environ.get("HF_TOKEN", "")

    from huggingface_hub import HfApi
    api = HfApi(token=token) if token else None

    _ensure_source_tokenizer()
    ckpt_vol.reload()

    # THE pinned probe set. This constant is the entire point of the harness.
    python = f"{VENV}/bin/python"
    print(f"[probe] ensuring the PINNED {PROBE_DATA_SHARDS}-shard probe set...")
    _run_streamed(f"{python} -m nanochat.dataset -n {PROBE_DATA_SHARDS}")

    model_path, used_step, origin = _resolve_checkpoint(
        api, token, preset["repo"], preset["run_tag"], step
    )
    trainer_module = _install_probe_trainer(need_beta=preset["hmap_beta"] > 0.0)

    label = f"{arm}-step{used_step}"
    probe_tag = f"icl-probe-{label}"
    logs_dir = Path(CACHE_DIR) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{probe_tag}.log"
    log_path.unlink(missing_ok=True)

    if num_gpus > 1:
        launcher = (f"{VENV}/bin/torchrun --standalone "
                    f"--nproc_per_node={num_gpus} -m {trainer_module} --")
    else:
        launcher = f"{python} -m {trainer_module}"

    variant_flags = f"--attn-variant={preset['attn_variant']} "
    if preset["attn_variant"] == "hmap":
        variant_flags += f"--hmap-alpha={preset['hmap_alpha']} "
        if preset["hmap_beta"] > 0.0:
            variant_flags += f"--hmap-beta={preset['hmap_beta']} "

    # ONE step, ALL learning rates zero: the step-0 evals (val bpb + ICL/
    # induction probe) fire on the loaded weights exactly; the single update
    # is a no-op. save-every is huge so nothing is checkpointed; the tiny
    # total-batch-size keeps the wasted step to one micro-batch per rank.
    cmd = (
        f"{launcher} "
        f"--depth=32 "
        f"--model-tag={probe_tag} "
        f"--num-iterations=1 "
        f"--save-every=1000000 "
        f"--window-pattern=L "
        f"{variant_flags}"
        f"--device-batch-size=4 "
        f"--total-batch-size={num_gpus * 4 * 2048} "
        f"--embedding-lr=0.0 --unembedding-lr=0.0 --matrix-lr=0.0 "
        f"--scalar-lr=0.0 --weight-decay=0.0 "
        f"--eval-every=1 --core-metric-every=-1 --sample-every=1000000 "
        f"--init-from-model={model_path} "
        f"{extra_args}"
    )

    print(f"[probe] arm={arm} origin={origin} operator={preset['attn_variant']} "
          f"a={preset['hmap_alpha']} b={preset['hmap_beta']} "
          f"shards={PROBE_DATA_SHARDS}")
    _run_streamed(cmd, log_path=log_path)

    # Parse the step-0 probe lines
    text = log_path.read_text()
    icl_rows = [m for m in ICL_RE.finditer(text)]
    bpb_rows = [m for m in BPB_RE.finditer(text)]
    assert icl_rows, "[probe] no ICL probe line found in the run log"
    m = icl_rows[0]
    result = {
        "arm": arm,
        "origin": origin,
        "checkpoint_step": used_step,
        "operator": {
            "attn_variant": preset["attn_variant"],
            "hmap_alpha": preset["hmap_alpha"],
            "hmap_beta": preset["hmap_beta"],
        },
        "probe_data_shards": PROBE_DATA_SHARDS,
        "icl_early": float(m.group(2)),
        "icl_late": float(m.group(3)),
        "icl_score": float(m.group(4)),
        "induction_loss": float(m.group(5)),
        "induction_acc": float(m.group(6)),
        "random_half_loss": float(m.group(7)),
        "val_bpb": float(bpb_rows[0].group(2)) if bpb_rows else None,
    }
    out_path = logs_dir / f"{probe_tag}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[probe] RESULT {label}: score={result['icl_score']:+.4f} "
          f"early={result['icl_early']:.4f} late={result['icl_late']:.4f} "
          f"induction_acc={result['induction_acc']:.4f} "
          f"bpb={result['val_bpb']}")

    # Scrub any scratch checkpoint dir the probe run may have created
    scratch = Path(CACHE_DIR) / "base_checkpoints" / probe_tag
    if scratch.exists():
        shutil.rmtree(scratch)
        print(f"[probe] removed scratch dir {scratch}")
    ckpt_vol.commit()

    if api is not None:
        for p, dest in [(out_path, f"probes/{probe_tag}.json"),
                        (log_path, f"probes/{probe_tag}.log")]:
            api.upload_file(path_or_fileobj=str(p), path_in_repo=dest,
                            repo_id=push_repo)
        print(f"[probe] archived -> {push_repo}/probes/")
    print("[probe] all done, returning.")
    return result


@app.local_entrypoint()
def main(
    arm: str = "amap",
    step: int = -1,
    num_gpus: int = 1,
    repo: str = "",
    run_tag: str = "",
    attn_variant: str = "",
    hmap_alpha: float = -1.0,
    hmap_beta: float = -1.0,
    push_repo: str = "sparsetrace/d32ft",
    extra_args: str = "",
):
    result = run_probe.remote(
        arm=arm, step=step, num_gpus=num_gpus, repo=repo, run_tag=run_tag,
        attn_variant=attn_variant, hmap_alpha=hmap_alpha, hmap_beta=hmap_beta,
        push_repo=push_repo, extra_args=extra_args,
    )
    print(json.dumps(result, indent=2))
