from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal
import common


app = modal.App("nanochat-diagnostics")
image = common.image.add_local_python_source("common")


# Run nanochat evaluation inside the repo venv baked into the Modal image.
# Diagnostics intentionally do NOT invoke scripts.base_train and do NOT
# torch.compile() the model. Measurement code should stay separate from training.
INNER = r"""
from __future__ import annotations

import json
import random
import sys

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from nanochat.icl_eval import icl_score, induction_score


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if torch.is_tensor(x):
        if x.numel() == 1:
            return x.item()
        return x.detach().cpu().tolist()
    try:
        import numpy as np
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
    except Exception:
        pass
    return x


def neutralize_missing_vintage_parameters(model, missing):
    # Public vintage checkpoints may omit only explicitly neutralizable
    # parameters introduced by the newer fork.
    with torch.no_grad():
        named = dict(model.named_parameters())
        for name in missing:
            p = named.get(name)
            if p is None:
                continue
            if (
                "value_embed" in name
                or "ve_gate" in name
                or "x0_lambda" in name
                or "smear_gate" in name
                or "smear_lambda" in name
                or "backout_lambda" in name
            ):
                p.zero_()
            elif "resid_lambda" in name:
                p.fill_(1.0)


cfg = json.load(open(sys.argv[1]))
meta = json.load(open(cfg["meta_path"]))
mc = dict(meta["model_config"])

# Backward compatibility for older HMAP checkpoints. Some early d12 metadata
# serialized newly-added HMAP fields as null/None even though training used the
# parser defaults. Passing None into the current GPTConfig can poison the HMAP
# arithmetic and trigger an asynchronous CUDA device-side assert.
if mc.get("attn_variant") == "hmap":
    if mc.get("hmap_alpha") is None:
        mc["hmap_alpha"] = 0.0
    if mc.get("hmap_beta") is None:
        mc["hmap_beta"] = 0.0
    if mc.get("witten") is None:
        mc["witten"] = False

if cfg.get("operator_override"):
    mc["attn_variant"] = cfg["operator_override"]

device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cuda":
    raise RuntimeError("d32 diagnostics require CUDA")

torch.manual_seed(cfg["seed"])
random.seed(cfg["seed"])

print(
    f"[diag] device={device} gpu={torch.cuda.get_device_name(0)}",
    flush=True,
)
print(
    f"[diag] building model: attn_variant={mc.get('attn_variant')} "
    f"alpha={mc.get('hmap_alpha')} beta={mc.get('hmap_beta')}",
    flush=True,
)

# Build on meta first so we do not initialize a duplicate full-size d32 model.
with torch.device("meta"):
    model = GPT(GPTConfig(**mc))
model.to_empty(device=device)

# Critical: initialize nanochat runtime buffers (e.g. RoPE tables) before
# checkpoint assignment. The checkpoint replaces trainable parameters, but
# non-persistent/runtime buffers are not guaranteed to come from state_dict.
model.init_weights()

state = torch.load(
    cfg["model_path"],
    map_location=device,
    weights_only=True,
)

own = model.state_dict()
missing = sorted(set(own) - set(state))
unexpected = sorted(set(state) - set(own))
shape_bad = sorted(
    k for k in set(own) & set(state)
    if tuple(own[k].shape) != tuple(state[k].shape)
)

assert not unexpected, f"unexpected keys: {unexpected[:10]}"
assert not shape_bad, f"shape mismatch: {shape_bad[:10]}"

ALLOW = (
    "value_embed",
    "ve_gate",
    "resid_lambda",
    "x0_lambda",
    "smear_gate",
    "smear_lambda",
    "backout_lambda",
)
bad = [k for k in missing if not any(a in k for a in ALLOW)]
assert not bad, f"missing outside vintage allowlist: {bad[:10]}"

# assign=True preserves checkpoint dtype/storage for matched parameters.
model.load_state_dict(state, strict=False, assign=True)
del state
neutralize_missing_vintage_parameters(model, missing)
model.eval()

tok = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tok.get_vocab_size()

sequence_len = int(
    mc.get("sequence_len")
    or mc.get("max_seq_len")
    or cfg["max_seq_len"]
)


def build_val_loader():
    return tokenizing_distributed_data_loader_bos_bestfit(
        tok,
        cfg["device_batch_size"],
        sequence_len,
        split="val",
        device=device,
    )


param_total = int(sum(p.numel() for p in model.parameters()))
param_trainable = int(
    sum(p.numel() for p in model.parameters() if p.requires_grad)
)

out = {
    "checkpoint_step": meta.get("step"),
    "model_config": {
        "n_layer": mc.get("n_layer"),
        "n_head": mc.get("n_head"),
        "n_embd": mc.get("n_embd"),
        "sequence_len": sequence_len,
        "attn_variant": mc.get("attn_variant"),
        "hmap_alpha": mc.get("hmap_alpha"),
        "hmap_beta": mc.get("hmap_beta"),
        "window_pattern": mc.get("window_pattern"),
    },
    "parameters": {
        "total": param_total,
        "trainable": param_trainable,
    },
    "vintage_missing_parameters": missing,
    "evaluation": {
        "device_batch_size": cfg["device_batch_size"],
        "icl_num_batches": cfg["icl_num_batches"],
        "eval_tokens_requested": cfg["eval_tokens"],
    },
}

print(f"[diag] parameters={param_total:,}", flush=True)

# -------------------------------------------------------------------------
# Validation bpb: direct nanochat evaluator, no optimizer or training loop.
if cfg["do_bpb"]:
    tokens_per_eval_step = cfg["device_batch_size"] * sequence_len
    eval_steps = max(1, cfg["eval_tokens"] // tokens_per_eval_step)
    actual_eval_tokens = eval_steps * tokens_per_eval_step

    print(
        f"[diag] val bpb: steps={eval_steps} "
        f"tokens≈{actual_eval_tokens:,} batch={cfg['device_batch_size']}",
        flush=True,
    )

    with torch.inference_mode():
        val_bpb = evaluate_bpb(
            model,
            build_val_loader(),
            eval_steps,
            token_bytes,
        )

    out["validation"] = {
        "bpb": float(val_bpb),
        "eval_steps": int(eval_steps),
        "eval_tokens": int(actual_eval_tokens),
    }
    print(f"[diag] Validation bpb={val_bpb:.6f}", flush=True)

# -------------------------------------------------------------------------
# ICL + induction: exact evaluator functions used by scripts.base_train.
# base_train used a compiled model for ICL as an optimization only. Here we
# intentionally stay uncompiled to avoid the HMAP compile stall.
if cfg["do_icl"]:
    print(
        f"[diag] ICL: num_batches={cfg['icl_num_batches']} "
        f"batch={cfg['device_batch_size']} (uncompiled)",
        flush=True,
    )

    with torch.inference_mode():
        icl = icl_score(
            model,
            build_val_loader(),
            num_batches=cfg["icl_num_batches"],
        )
        ind = induction_score(
            model,
            vocab_size,
            device,
        )

    out["icl"] = {
        "early": float(icl["early"]),
        "late": float(icl["late"]),
        "score": float(icl["score"]),
    }
    out["induction"] = {
        "loss": float(ind["induction_loss"]),
        "acc": float(ind["induction_acc"]),
        "random_half_loss": float(ind["random_half_loss"]),
    }

    print(
        f"[diag] ICL early={icl['early']:.4f} "
        f"late={icl['late']:.4f} "
        f"score={icl['score']:+.4f}",
        flush=True,
    )
    print(
        f"[diag] Induction loss={ind['induction_loss']:.4f} "
        f"acc={ind['induction_acc']:.4f} "
        f"random-half={ind['random_half_loss']:.4f}",
        flush=True,
    )

# -------------------------------------------------------------------------
# CORE: same direct evaluator used in scripts.base_train.
if cfg["do_core"]:
    from scripts.base_eval import evaluate_core

    print(
        f"[diag] CORE max_per_task={cfg['max_per_task']}",
        flush=True,
    )

    with torch.inference_mode():
        core = evaluate_core(
            model,
            tok,
            device,
            max_per_task=cfg["max_per_task"],
        )

    out["core"] = {
        "metric": float(core["core_metric"]),
        "centered_results": jsonable(core["centered_results"]),
    }
    print(f"[diag] CORE={core['core_metric']:.6f}", flush=True)

# -------------------------------------------------------------------------
# Sampling. HMAP has no KV-cache Engine path, so model.generate is used
# directly for every operator.
if cfg["do_sample"]:
    print("[diag] sampling", flush=True)
    samples = []

    for prompt in cfg["prompts"]:
        for i in range(cfg["num_samples"]):
            tokens = tok(prompt, prepend="<|bos|>")
            with torch.inference_mode():
                gen = list(
                    model.generate(
                        tokens,
                        max_tokens=cfg["max_tokens"],
                        temperature=cfg["temperature"],
                        top_k=cfg["top_k"],
                        seed=cfg["seed"] + i,
                    )
                )

            text = tok.decode(tokens + gen)
            print(text, flush=True)
            samples.append(
                {
                    "prompt": prompt,
                    "seed": cfg["seed"] + i,
                    "text": text,
                }
            )

    out["samples"] = samples

out["peak_cuda_memory_mib"] = float(
    torch.cuda.max_memory_allocated() / 1024 / 1024
)

with open(cfg["out_path"], "w") as f:
    json.dump(jsonable(out), f, indent=2)

print(
    f"[diag] done; peak_cuda_memory="
    f"{out['peak_cuda_memory_mib']:.1f} MiB",
    flush=True,
)
"""


@app.function(
    image=image,
    gpu="H100",
    cpu=16,
    memory=65536,
    timeout=6 * 60 * 60,
    scaledown_window=5,
    volumes={common.CACHE_DIR: common.ckpt_vol},
    secrets=[common.hf_secret],
)
def diagnose(
    state: str = "amap2",
    repo: str = "",
    run_tag: str = "",
    step: int = -1,
    operator_override: str = "",
    do_bpb: bool = True,
    do_icl: bool = True,
    do_core: bool = True,
    do_sample: bool = True,
    # Direct diagnostics use 2^19 validation tokens by default. This avoids
    # accidentally inheriting base_train's huge 80*2^19-token validation sweep.
    eval_tokens: int = 524288,
    # Canonical base_train ICL call.
    icl_num_batches: int = 8,
    # Conservative for eager HMAP/AMAP at sequence length 2048.
    device_batch_size: int = 1,
    max_seq_len: int = 2048,
    max_per_task: int = 500,
    num_samples: int = 1,
    max_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 50,
    seed: int = 42,
    results_repo: str = common.RESULTS_REPO,
):
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN required")

    api = HfApi(token=token)

    common.ensure_source_tokenizer()
    common.ckpt_vol.reload()
    info = common.resolve_state(
        api,
        token,
        state,
        repo,
        run_tag,
        step,
    )

    meta = json.loads(Path(info["meta_path"]).read_text())
    meta_mc = dict(meta["model_config"])

    op = (
        operator_override
        or meta_mc.get("attn_variant")
        or info.get("operator_hint")
        or "unknown"
    )
    label = f"{state}-step{info['step']}-{op}"

    print(
        f"[diag] resolved state={state} -> {info['origin']} "
        f"(step={info['step']}, operator={op})",
        flush=True,
    )

    wd = Path("/tmp/diagnostics")
    wd.mkdir(parents=True, exist_ok=True)

    out_path = wd / f"{label}.json"
    cfg_path = wd / "job.json"
    inner_path = wd / "inner.py"

    cfg = {
        "meta_path": str(info["meta_path"]),
        "model_path": str(info["model_path"]),
        "operator_override": operator_override,
        "do_bpb": do_bpb,
        "do_icl": do_icl,
        "do_core": do_core,
        "do_sample": do_sample,
        "eval_tokens": eval_tokens,
        "icl_num_batches": icl_num_batches,
        "device_batch_size": device_batch_size,
        "max_seq_len": max_seq_len,
        "max_per_task": max_per_task,
        "num_samples": num_samples,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "seed": seed,
        "out_path": str(out_path),
        "prompts": [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ],
    }

    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    inner_path.write_text(INNER)

    subprocess.run(
        (
            f"cd {common.REPO_DIR} && "
            f"PYTHONPATH={common.REPO_DIR} "
            f"{common.VENV}/bin/python -u "
            f"{inner_path} {cfg_path}"
        ),
        shell=True,
        check=True,
        executable="/bin/bash",
    )

    result = json.loads(out_path.read_text())

    result["state"] = state
    result["origin"] = info["origin"]
    result["repo"] = info["repo"]
    result["run_tag"] = info["run_tag"]
    result["resolved_step"] = info["step"]
    result["operator"] = op
    result["diagnostics_protocol"] = {
        "direct_eval": True,
        "torch_compile": False,
        "pseudo_training": False,
        "eval_tokens": eval_tokens,
        "icl_num_batches": icl_num_batches,
        "device_batch_size": device_batch_size,
    }

    out_path.write_text(
        json.dumps(result, indent=2) + "\n"
    )

    common.upload_json(
        api,
        out_path,
        label,
        "diagnostics",
        results_repo,
    )

    return result


@app.local_entrypoint()
def main(
    state: str = "amap2",
    repo: str = "",
    run_tag: str = "",
    step: int = -1,
    operator_override: str = "",
    do_bpb: bool = True,
    do_icl: bool = True,
    do_core: bool = True,
    do_sample: bool = True,
    eval_tokens: int = 524288,
    icl_num_batches: int = 8,
    device_batch_size: int = 1,
    max_seq_len: int = 2048,
    max_per_task: int = 500,
    num_samples: int = 1,
    max_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 50,
    seed: int = 42,
    results_repo: str = common.RESULTS_REPO,
):
    r = diagnose.remote(
        state=state,
        repo=repo,
        run_tag=run_tag,
        step=step,
        operator_override=operator_override,
        do_bpb=do_bpb,
        do_icl=do_icl,
        do_core=do_core,
        do_sample=do_sample,
        eval_tokens=eval_tokens,
        icl_num_batches=icl_num_batches,
        device_batch_size=device_batch_size,
        max_seq_len=max_seq_len,
        max_per_task=max_per_task,
        num_samples=num_samples,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
        results_repo=results_repo,
    )

    print(json.dumps(r, indent=2))
