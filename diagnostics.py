from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
import modal
import common

app = modal.App("nanochat-diagnostics")
image = common.image.add_local_python_source("common")

INNER = r"""
import json, sys, torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer

cfg = json.load(open(sys.argv[1]))
meta = json.load(open(cfg["meta_path"]))
mc = dict(meta["model_config"])
if cfg.get("operator_override"):
    mc["attn_variant"] = cfg["operator_override"]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT(GPTConfig(**mc)).to(device)
state = torch.load(cfg["model_path"], map_location=device, weights_only=True)

own = model.state_dict()
missing = sorted(set(own) - set(state))
unexpected = sorted(set(state) - set(own))
shape_bad = [k for k in set(own) & set(state) if tuple(own[k].shape) != tuple(state[k].shape)]
assert not unexpected, f"unexpected keys: {unexpected[:5]}"
assert not shape_bad, f"shape mismatch: {shape_bad[:5]}"

ALLOW = ("value_embed","ve_gate","resid_lambda","x0_lambda","smear_gate","smear_lambda","backout_lambda")
bad = [k for k in missing if not any(a in k for a in ALLOW)]
assert not bad, f"missing outside vintage allowlist: {bad[:5]}"

model.load_state_dict(state, strict=False)
with torch.no_grad():
    for k,p in model.named_parameters():
        if k not in missing:
            continue
        if "value_embeds" in k:
            p.zero_()
        elif "resid_lambda" in k:
            p.fill_(1.0)
        elif "x0_lambda" in k or "backout_lambda" in k:
            p.zero_()
model.eval()
tok = get_tokenizer()

out = {
    "checkpoint_step": meta.get("step"),
    "model_config": {
        "depth": mc.get("depth"),
        "attn_variant": mc.get("attn_variant"),
        "hmap_alpha": mc.get("hmap_alpha"),
        "hmap_beta": mc.get("hmap_beta"),
        "window_pattern": mc.get("window_pattern"),
    },
    "parameters": {
        "total": int(sum(p.numel() for p in model.parameters())),
        "trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    },
}

if cfg["do_core"]:
    from scripts.base_eval import evaluate_core
    core = evaluate_core(model, tok, device, max_per_task=cfg["max_per_task"])
    out["core"] = {
        "metric": float(core["core_metric"]),
        "centered_results": core["centered_results"],
    }
    print(f"[diag] CORE={core['core_metric']:.6f}", flush=True)

if cfg["do_sample"]:
    samples = []
    for prompt in cfg["prompts"]:
        for i in range(cfg["num_samples"]):
            tokens = tok(prompt, prepend="<|bos|>")
            with torch.no_grad():
                gen = list(model.generate(
                    tokens, max_tokens=cfg["max_tokens"],
                    temperature=cfg["temperature"], top_k=cfg["top_k"],
                    seed=cfg["seed"] + i
                ))
            text = tok.decode(tokens + gen)
            print(text, flush=True)
            samples.append({"prompt": prompt, "seed": cfg["seed"]+i, "text": text})
    out["samples"] = samples

with open(cfg["out_path"], "w") as f:
    json.dump(out, f, indent=2)
"""

@app.function(
    image=image,
    gpu="H100",
    cpu=16,
    memory=65536,
    timeout=6*60*60,
    scaledown_window=5,
    volumes={common.CACHE_DIR: common.ckpt_vol},
    secrets=[common.hf_secret],
)
def diagnose(
    state: str="amap2",
    repo: str="",
    run_tag: str="",
    step: int=-1,
    operator_override: str="",
    do_core: bool=True,
    do_sample: bool=True,
    max_per_task: int=500,
    num_samples: int=1,
    max_tokens: int=64,
    temperature: float=0.8,
    top_k: int=50,
    seed: int=42,
    results_repo: str=common.RESULTS_REPO,
):
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN","")
    if not token:
        raise RuntimeError("HF_TOKEN required")
    api = HfApi(token=token)

    common.ensure_source_tokenizer()
    common.ckpt_vol.reload()
    info = common.resolve_state(api, token, state, repo, run_tag, step)

    meta = json.loads(Path(info["meta_path"]).read_text())
    op = operator_override or meta["model_config"].get("attn_variant") or info.get("operator_hint") or "unknown"
    label = f"{state}-step{info['step']}-{op}"

    wd = Path("/tmp/diagnostics")
    wd.mkdir(parents=True, exist_ok=True)
    out_path = wd / f"{label}.json"
    cfg = {
        "meta_path": str(info["meta_path"]),
        "model_path": str(info["model_path"]),
        "operator_override": operator_override,
        "do_core": do_core,
        "do_sample": do_sample,
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
    (wd/"job.json").write_text(json.dumps(cfg))
    (wd/"inner.py").write_text(INNER)
    subprocess.run(
        f"cd {common.REPO_DIR} && PYTHONPATH={common.REPO_DIR} "
        f"{common.VENV}/bin/python -u {wd}/inner.py {wd}/job.json",
        shell=True, check=True, executable="/bin/bash"
    )

    result = json.loads(out_path.read_text())
    result["state"] = state
    result["origin"] = info["origin"]
    result["repo"] = info["repo"]
    result["run_tag"] = info["run_tag"]
    result["resolved_step"] = info["step"]
    out_path.write_text(json.dumps(result, indent=2)+"\n")
    common.upload_json(api, out_path, label, "diagnostics", results_repo)
    return result

@app.local_entrypoint()
def main(
    state: str="amap2",
    repo: str="",
    run_tag: str="",
    step: int=-1,
    operator_override: str="",
    do_core: bool=True,
    do_sample: bool=True,
    max_per_task: int=500,
    num_samples: int=1,
    max_tokens: int=64,
    temperature: float=0.8,
    top_k: int=50,
    seed: int=42,
    results_repo: str=common.RESULTS_REPO,
):
    r = diagnose.remote(
        state=state, repo=repo, run_tag=run_tag, step=step,
        operator_override=operator_override, do_core=do_core, do_sample=do_sample,
        max_per_task=max_per_task, num_samples=num_samples, max_tokens=max_tokens,
        temperature=temperature, top_k=top_k, seed=seed, results_repo=results_repo
    )
    print(json.dumps(r, indent=2))
