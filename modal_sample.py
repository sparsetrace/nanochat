"""
modal_sample.py — sample text from a trained nanochat checkpoint on an
ephemeral Modal GPU, straight from the HuggingFace repo.

Lives in the repo root next to modal_train.py (it reuses that file's image and
Volume definitions, so the two stay consistent automatically).

What it does
------------
1. Downloads a checkpoint (model_*.pt + meta_*.json) from
   <hf_repo>/<hf_subfolder>/latest/  or  .../checkpoints/<run_tag>/
2. Rebuilds the exact model from the checkpoint's own saved model_config
   (attn_variant, hmap_alpha, window_pattern, ... all come from the meta —
   nothing to specify manually, HMAP checkpoints just work via naive generate).
3. Generates num_samples continuations per prompt and prints them.
4. Uploads the samples to <hf_subfolder>/samples/<run_tag>_step<N>_sampled.txt

Usage
-----
    modal run modal_sample.py --hf-subfolder AMAP --run-tag d12-a00
    modal run modal_sample.py --hf-subfolder attention --run-tag d12-base \
        --prompts "Once upon a time;The meaning of life is" \
        --num-samples 3 --max-tokens 64 --temperature 0.8

Runs attached (a sampling job is minutes), so output appears directly in the
launching terminal / GitHub Actions log.
"""

import json
import os
import re
import subprocess
from pathlib import Path

import modal

# Reuse the training harness's image + volume so environments never diverge.
# NOTE: the container re-imports this file at startup, so it must also be able
# to import modal_train there — add_local_python_source ships that file's
# source with the app (it is otherwise excluded from the baked image).
import modal_train as _train

ckpt_vol = _train.ckpt_vol
CACHE_DIR = _train.CACHE_DIR
REPO_DIR = _train.REPO_DIR
VENV = _train.VENV
HF_REPO_DEFAULT = _train.HF_REPO_DEFAULT
image = _train.image.add_local_python_source("modal_train")

app = modal.App("nanochat-sample")

hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})

# The in-container generation script. Runs under the repo venv (torch + nanochat).
# Model is rebuilt from the checkpoint's own meta: no manual architecture flags.
SAMPLE_PY = r'''
import json, sys, torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer

cfg = json.load(open(sys.argv[1]))          # job config written by the harness
meta = json.load(open(cfg["meta_path"]))
model_config = meta["model_config"]
print(f"[sample] checkpoint step {meta['step']}, model_config: "
      f"attn_variant={model_config.get('attn_variant')} "
      f"hmap_alpha={model_config.get('hmap_alpha')} "
      f"window_pattern={model_config.get('window_pattern')}")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT(GPTConfig(**model_config)).to(device)
state = torch.load(cfg["model_path"], map_location=device)
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
    f.write(f"# samples from step {meta['step']} "
            f"(temperature={cfg['temperature']}, top_k={cfg['top_k']}, "
            f"max_tokens={cfg['max_tokens']})\n")
    f.write("\n".join(lines) + "\n")
print(f"[sample] wrote {len(lines)} samples -> {cfg['out_path']}")
'''


@app.function(
    image=image,
    gpu="A10G",            # sampling a d12 needs little; ~cents per job
    cpu=4,
    memory=16384,
    timeout=30 * 60,
    scaledown_window=5,
    volumes={CACHE_DIR: ckpt_vol},   # for the tokenizer (trained once, on the Volume)
    secrets=[hf_secret],
)
def sample_checkpoint(
    hf_repo: str = HF_REPO_DEFAULT,
    hf_subfolder: str = "attention",
    run_tag: str = "d12-base",
    source: str = "latest",      # "latest" -> <sub>/latest/ ; "archive" -> <sub>/checkpoints/<run_tag>/
    step: int = -1,              # -1 = newest step found at the source
    prompts: str = "",           # semicolon-separated; empty = default 7 prompts
    num_samples: int = 1,
    max_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 50,
    seed: int = 42,
) -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    token = os.environ.get("HF_TOKEN", "")
    assert hf_repo and token, "hf_repo and HF_TOKEN are required"
    api = HfApi(token=token)

    prefix = (f"{hf_subfolder}/latest/" if source == "latest"
              else f"{hf_subfolder}/checkpoints/{run_tag}/")
    files = [f for f in api.list_repo_files(repo_id=hf_repo) if f.startswith(prefix)]
    steps = sorted({int(m.group(1)) for f in files
                    if (m := re.match(r"model_(\d+)\.pt$", Path(f).name))})
    assert steps, f"no model_*.pt found under {hf_repo}/{prefix}"
    use_step = step if step > 0 else steps[-1]
    assert use_step in steps, f"step {use_step} not found; available: {steps}"

    workdir = Path("/tmp/sample")
    workdir.mkdir(parents=True, exist_ok=True)
    model_path = hf_hub_download(repo_id=hf_repo, token=token,
                                 filename=f"{prefix}model_{use_step:06d}.pt",
                                 local_dir=str(workdir))
    meta_path = hf_hub_download(repo_id=hf_repo, token=token,
                                filename=f"{prefix}meta_{use_step:06d}.json",
                                local_dir=str(workdir))

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

    out_path = workdir / f"{run_tag}_step{use_step}_sampled.txt"
    job = {
        "meta_path": meta_path, "model_path": model_path,
        "prompts": prompt_list, "num_samples": num_samples,
        "max_tokens": max_tokens, "temperature": temperature,
        "top_k": top_k, "seed": seed, "out_path": str(out_path),
    }
    (workdir / "job.json").write_text(json.dumps(job))
    (workdir / "sample_script.py").write_text(SAMPLE_PY)

    print(f"[sample] {hf_repo}/{prefix} step {use_step} | "
          f"{len(prompt_list)} prompts x {num_samples} samples x {max_tokens} tokens")
    subprocess.run(
        f"cd {REPO_DIR} && PYTHONPATH={REPO_DIR} "
        f"{VENV}/bin/python -u {workdir}/sample_script.py {workdir}/job.json",
        shell=True, check=True, executable="/bin/bash",
    )

    dest = f"{hf_subfolder}/samples/{run_tag}_step{use_step}_sampled.txt"
    api.upload_file(path_or_fileobj=str(out_path), path_in_repo=dest, repo_id=hf_repo)
    print(f"[sample] uploaded -> {hf_repo}/{dest}")
    return {"ok": True, "step": use_step, "uploaded": dest}


@app.local_entrypoint()
def main(
    hf_repo: str = HF_REPO_DEFAULT,
    hf_subfolder: str = "attention",
    run_tag: str = "d12-base",
    source: str = "latest",
    step: int = -1,
    prompts: str = "",
    num_samples: int = 1,
    max_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 50,
    seed: int = 42,
):
    result = sample_checkpoint.remote(
        hf_repo=hf_repo, hf_subfolder=hf_subfolder, run_tag=run_tag,
        source=source, step=step, prompts=prompts, num_samples=num_samples,
        max_tokens=max_tokens, temperature=temperature, top_k=top_k, seed=seed,
    )
    print(result)
