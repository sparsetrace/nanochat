"""
XI_sim.py — frozen anti-attention (xi) scan: NO training, one GPU job.

    modal run XI_sim.py                     # AMAP checkpoint, xi = 0.0 .. 1.0
    modal run XI_sim.py --arm regrafted     # same ray from the standard corner

Loads one checkpoint ONCE, then sweeps the mirror-mix coordinate

    s_xi = s + (1 - xi) * s^T  =  (2 - xi) * [symmetric part] + xi * A

    xi = 1.0 : the checkpoint's own operator (AMAP for arm=amap) — baseline
    xi = 0.0 : pure symmetrized kernel at doubled coupling — bare DMAP,
               undirected, no tilt

evaluating the identifiable synthetic induction probe (random half + exact
repeat) at every point. The scan is PAIRED: one fixed token batch is reused
across all xi, so curves differ only through the operator.

Three rays through the two-coupling (c_sym, c_flux) plane (the baseline
(1,1) sits on all three — free replicates):
  xi        : (2-t, t)  — the anti-attention diagonal (metric heats AND flux
              anneals; the physical path)
  fluxcut   : (1, t)    — pinned metric, flux -> 0. The exact part of the
              antisymmetric sector washes under row-softmax, so this ray is
              the FROZEN COEXACT DIAL.
  symheat   : (c, 1)    — pinned flux, metric coupling 1 -> 2 (temperature /
              PSD-diagonal self-lock control; c=2 at flux 1 is close to the
              uniform-doubling point that collapsed frozen induction)
plus a fourth, extending the master curve past its trained boundary:
  fluxboost : (1, t), t in (1, 2] — pinned metric, flux OVER-driven (r > 1).
              Decides whether calibrated induction beyond the trained
              coupling improves, saturates, or declines — i.e. whether the
              trained model sits at, below, or above its own optimal
              relative directed coupling.
Calibrated accuracies from all rays are expected to collapse onto a master
curve in the gauge-invariant ratio r = c_flux/c_sym (per-head temperatures
span the radial/temperature direction, so calibration quotients it out).
The xi-cliff is interpretable only against the other two rays: fluxcut vs
symheat decides whether the collapse is flux loss or metric heating.

CALIBRATION RUNG (calib_steps > 0, default 40): at every point, after the
frozen evaluation, a scalars-only recalibration runs on a HELD-OUT batch —
Adam on the conditioning parameters (resid/x0 lambdas, smear/backout,
smear/ve gates) plus fresh per-head logit temperatures, ALL MATRICES FROZEN.
These parameters can repair sharpness/gain/mixing but cannot re-aim an
attention edge and cannot move the ratio c_flux/c_sym (both couplings
multiply parts of the same pre-softmax tensor). Whatever induction the
calib row recovers over the frozen row was conditioning; whatever it cannot
recover is not repairable without touching circuits. Scalars are restored
from the checkpoint before each point — no leakage across points. The
calibrated BASELINE (1,1) is the control: it should roughly match frozen.

The mirror mix is applied INSIDE the eager hmap block BEFORE the causal mask
(transposing after the mask would drag -inf from the invisible triangle onto
visible entries) via an ephemeral copy of gpt.py — the checked-in file is
untouched. Grad is left enabled during the forward so gpt.py takes the full
(B,H,T,T) path where the transpose is well-defined (the chunked eval path
processes row blocks, which cannot be mirror-mixed); outputs are detached
immediately and no backward is run.

Sanity built in: XI_MIX=None (hook off) must match xi=1.0/raw exactly.

Arm presets (operator of the loaded checkpoint):
  amap      : sparsetrace/d32ft / d32ft-amap      -> (beta, alpha) = (0, 0)
  regrafted : sparsetrace/amap2nanochat           -> (1, 0)
  original  : karpathy/nanochat-d32 (vintage)     -> (1, 0)
  cmap      : sparsetrace/d32cmap                 -> (1, 1)

Outputs: JSON + CSV (xi, variant, induction_loss, induction_acc,
random_half_loss), uploaded to <push_repo>/probes/.

IMPORTANT harness constraint: this function runs under the container's
SYSTEM python (modal + huggingface_hub only). torch lives in the repo venv.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import modal

app = modal.App("nanochat-xi-sim")

KARPATHY_MODEL = "model_000650.pt"
KARPATHY_META = "meta_000650.json"
D32_CONFIG = dict(
    sequence_len=2048, vocab_size=65536, n_layer=32, n_head=16,
    n_kv_head=16, n_embd=2048,
)

ARMS = {
    "amap":      dict(repo="sparsetrace/d32ft", run_tag="d32ft-amap",
                      hmap_alpha=0.0, hmap_beta=0.0),
    "regrafted": dict(repo="sparsetrace/amap2nanochat", run_tag="amap2nanochat",
                      hmap_alpha=0.0, hmap_beta=1.0),
    "original":  dict(repo="karpathy/nanochat-d32", run_tag=None,
                      hmap_alpha=0.0, hmap_beta=1.0),
    "cmap":      dict(repo="sparsetrace/d32cmap", run_tag="d32cmap",
                      hmap_alpha=1.0, hmap_beta=1.0),
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
            "icl_probe.py", "sector_probe.py", "XI_sim.py",
        ],
    )
    .run_commands(f"cd {REPO_DIR} && uv sync --extra gpu")
)


def _run_streamed(cmd: str):
    print(f"[xi] running: {cmd}", flush=True)
    proc = subprocess.Popen(
        cmd, shell=True, executable="/bin/bash", cwd=REPO_DIR,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {cmd}")


def _hf_download_into(repo_id: str, filename: str, dest: Path, token=None):
    from huggingface_hub import hf_hub_download
    cached = hf_hub_download(
        repo_id=repo_id, filename=filename, token=token,
        local_dir=str(Path(CACHE_DIR) / "hf_downloads" / repo_id.replace("/", "__")),
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, dest)
    return dest


def _weights_steps(files, prefix):
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


def _resolve_checkpoint(api, token, repo, run_tag, step):
    if run_tag is None:
        seed_dir = Path(CACHE_DIR) / "bootstrap" / "karpathy-d32"
        model = _hf_download_into(repo, KARPATHY_MODEL,
                                  seed_dir / KARPATHY_MODEL, token=None)
        meta = _hf_download_into(repo, KARPATHY_META,
                                 seed_dir / KARPATHY_META, token=None)
        return model, meta, 650, f"{repo}:{KARPATHY_MODEL}"

    ckpt_dir = Path(CACHE_DIR) / "base_checkpoints" / run_tag
    local = ([p.name for p in ckpt_dir.iterdir() if p.is_file()]
             if ckpt_dir.exists() else [])
    steps = _weights_steps(local, "")
    if steps and (step in steps or step < 0):
        use = step if step > 0 else steps[-1]
        print(f"[xi] checkpoint (volume): {run_tag} step {use}")
        return (ckpt_dir / f"model_{use:06d}.pt",
                ckpt_dir / f"meta_{use:06d}.json",
                use, f"volume:{run_tag}:step-{use}")

    files = list(api.list_repo_files(repo_id=repo))
    for prefix in (f"checkpoints/{run_tag}/", "latest/"):
        rsteps = _weights_steps(files, prefix)
        if not rsteps:
            continue
        use = step if (step > 0 and step in rsteps) else (rsteps[-1] if step < 0 else None)
        if use is None:
            continue
        dest = Path(CACHE_DIR) / "probe_seeds" / run_tag / f"{use:06d}"
        model = _hf_download_into(repo, f"{prefix}model_{use:06d}.pt",
                                  dest / f"model_{use:06d}.pt", token=token)
        meta = _hf_download_into(repo, f"{prefix}meta_{use:06d}.json",
                                 dest / f"meta_{use:06d}.json", token=token)
        print(f"[xi] checkpoint (HF): {repo}/{prefix} step {use}")
        return model, meta, use, f"{repo}:{prefix}step-{use}"
    raise RuntimeError(f"No checkpoint for {run_tag} (step {step})")


def _install_xi_gpt() -> str:
    """Ephemeral nanochat/_xi_sim_gpt.py: gpt.py + a mirror-mix hook applied
    to the PRE-MASK, PRE-SCALE logits of the eager hmap block. Anchor-
    verified; the checked-in gpt.py is untouched."""
    src = Path(REPO_DIR) / "nanochat" / "gpt.py"
    dst = Path(REPO_DIR) / "nanochat" / "_xi_sim_gpt.py"
    text = src.read_text()

    if "hmap_beta" not in text:
        raise RuntimeError(
            "nanochat/gpt.py has no hmap_beta — apply apply_cmap_patch.py "
            "first (the (1,0)==standard identity is what lets standard arms "
            "run through the eager branch)."
        )

    a1 = "_EAGER_MASK_CACHE = {}"
    assert text.count(a1) == 1, "mask-cache anchor drifted"
    text = text.replace(
        a1,
        a1 + "\n# xi-scan mirror mix: set to (temp_scale, mix) to apply\n"
             "# s -> temp_scale * (s + mix * s^T) on the pre-mask logits.\n"
             "XI_MIX = None\n"
             "# Calibration temperatures: list of per-layer (H,) tensors\n"
             "# multiplying the pre-mask logits (trainable during the\n"
             "# scalars-only calibration rung).\n"
             "XI_TEMP = None",
        1,
    )

    a2 = "                logits = logits * scale + mask[rows]                # broadcast over (B,H)\n"
    assert text.count(a2) == 1, "hmap-block anchor drifted"
    text = text.replace(
        a2,
        "                if XI_MIX is not None:\n"
        "                    _ts, _mix = XI_MIX\n"
        "                    logits = _ts * (logits + _mix * logits.transpose(-2, -1))\n"
        "                if XI_TEMP is not None:\n"
        "                    logits = logits * XI_TEMP[self.layer_idx].to(logits.dtype).view(1, -1, 1, 1)\n"
        + a2,
        1,
    )
    dst.write_text(text)
    print(f"[xi] installed mirror-mix module: {dst}")
    return "nanochat._xi_sim_gpt"


PROBE_PY = r'''
import csv, json, sys
import torch
import torch.nn.functional as F

cfg = json.load(open(sys.argv[1]))
import nanochat._xi_sim_gpt as G
import dataclasses

device = "cuda"
torch.manual_seed(cfg["seed"])
fields = {f.name for f in dataclasses.fields(G.GPTConfig)}

if cfg["vintage"]:
    template = dict(cfg["d32_config"])
else:
    template = json.load(open(cfg["meta_path"]))["model_config"]
for k, v in cfg["attn_overrides"].items():
    if k in fields:
        template[k] = v
    elif k != "hmap_h":
        raise RuntimeError(f"GPTConfig lacks required field {k}")
template["attn_variant"] = "hmap"
template["window_pattern"] = "L"
template = {k: v for k, v in template.items() if k in fields}
print("[xi-run] base operator: " + ", ".join(
    f"{k}={template.get(k)}" for k in
    ("attn_variant","hmap_alpha","hmap_beta","hmap_h") if k in template), flush=True)

model = G.GPT(G.GPTConfig(**template)).to(device)
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
if missing:
    print(f"[xi-run] graft-neutralized {len(missing)} vintage-absent params", flush=True)
model.eval()

# ── Calibration rung: scalars-only recalibration ────────────────────────────
# Trainable set = conditioning parameters only. These can fix sharpness,
# gain, and mixing ratios but CANNOT re-aim an attention edge, and CANNOT
# move the ratio c_flux/c_sym (both couplings multiply parts of the same
# pre-softmax tensor, and every parameter below rescales that tensor or the
# residual stream as a whole). Matrices stay frozen.
for p in model.parameters():
    p.requires_grad_(False)
scalar_params = {n: p for n, p in model.named_parameters()
                 if any(tag in n for tag in
                        ("resid_lambdas", "x0_lambdas", "smear_lambda",
                         "backout_lambda", "smear_gate", "ve_gate"))}
scalar_init = {n: p.detach().clone() for n, p in scalar_params.items()}
n_scalar = sum(p.numel() for p in scalar_params.values())
H = int(template["n_head"]); L = int(template["n_layer"])
print(f"[xi-run] calibration set: {len(scalar_params)} tensors, "
      f"{n_scalar} scalar params + {L*H} per-head temperatures", flush=True)

B, T = cfg["batch"], cfg["seqlen"]
V = int(template["vocab_size"])
half = T // 2
# ONE paired batch, reused at every xi
x = torch.randint(0, min(V, 50000), (B, half), device=device)
idx = torch.cat([x, x], dim=1)
tgt = idx[:, 1:]

# HELD-OUT calibration batch (different seed, never probed on)
gen = torch.Generator(device="cpu").manual_seed(cfg["calib_seed"])
Bc = cfg["calib_batch"]
xc = torch.randint(0, min(V, 50000), (Bc, half), generator=gen).to(device)
idx_c = torch.cat([xc, xc], dim=1)
tgt_c = idx_c[:, 1:]

def calibrate(steps, lr):
    """Fit scalars + fresh per-head temperatures on the held-out batch at
    the CURRENT (c_sym, c_flux). Leaves the calibrated state active for the
    probe evaluation; caller restores afterwards. Returns (first, last) CE."""
    temps = [torch.ones(H, device=device, requires_grad=True)
             for _ in range(L)]
    G.XI_TEMP = temps
    for p in scalar_params.values():
        p.requires_grad_(True)
    opt = torch.optim.Adam(list(scalar_params.values()) + temps, lr=lr)
    first = last = None
    for it in range(steps):
        opt.zero_grad(set_to_none=True)
        lg_c = model(idx_c)[:, :-1]
        loss = F.cross_entropy(lg_c.reshape(-1, lg_c.size(-1)),
                               tgt_c.reshape(-1))
        loss.backward()
        opt.step()
        if it == 0:
            first = loss.item()
        last = loss.item()
    for p in scalar_params.values():
        p.requires_grad_(False)
    return first, last

def restore_scalars():
    with torch.no_grad():
        for n, p in scalar_params.items():
            p.copy_(scalar_init[n])
    G.XI_TEMP = None

def evaluate():
    # grad ENABLED so the hmap branch takes the full (B,H,T,T) path where the
    # mirror transpose is well-defined; detach immediately, no backward.
    logits = model(idx).detach()
    lg = logits[:, :-1]
    ce = F.cross_entropy(lg.reshape(-1, lg.size(-1)), tgt.reshape(-1),
                         reduction="none").view(B, T - 1)
    pred = lg.argmax(-1)
    acc = (pred[:, half - 1 :] == tgt[:, half - 1 :]).float().mean().item()
    return (ce[:, half - 1 :].mean().item(),
            acc,
            ce[:, : half - 1].mean().item())

# Sanity: hook off must equal xi=1.0 raw (mix=0, ts=1) exactly
G.XI_MIX = None
base = evaluate()
G.XI_MIX = (1.0, 0.0)
same = evaluate()
assert abs(base[0] - same[0]) < 1e-6 and abs(base[2] - same[2]) < 1e-6, \
    f"hook identity failed: {base} vs {same}"
print(f"[xi-run] hook identity OK | baseline (xi=1.0): "
      f"induction_loss {base[0]:.4f} acc {base[1]:.4f} "
      f"random_half {base[2]:.4f}", flush=True)

def set_couplings(c_sym, c_flux):
    # hook applies ts * (s + mix * s^T); solve for target (c_sym, c_flux):
    #   ts*(1+mix) = c_sym,  ts*(1-mix) = c_flux
    ts = 0.5 * (c_sym + c_flux)
    mix = 0.0 if ts == 0.0 else (c_sym - c_flux) / (c_sym + c_flux)
    G.XI_MIX = (ts, mix)

# Three rays through the two-coupling (c_sym, c_flux) plane. (1,1) is the
# baseline and appears on every ray — free replicates. NOTE c_flux scales the
# whole antisymmetric sector, but the exact part washes under row-softmax, so
# the fluxcut ray is the FROZEN COEXACT DIAL at pinned metric coupling.
points = []
for i in range(11):
    t = round(0.1 * i, 1)
    points.append(("xi", t, 2.0 - t, t))        # anti-attention diagonal
    points.append(("fluxcut", t, 1.0, t))       # pinned metric, flux -> 0
for i in range(11):
    c = round(1.0 + 0.1 * i, 1)
    points.append(("symheat", c, c, 1.0))       # pinned flux, metric heats
for i in range(1, 11):
    t = round(1.0 + 0.1 * i, 1)
    points.append(("fluxboost", t, 1.0, t))     # pinned metric, flux OVER-driven
                                                # past its trained value (r > 1)

calib_steps = int(cfg["calib_steps"])
calib_lr = float(cfg["calib_lr"])
rows = []
for path, coord, cs, cf in points:
    set_couplings(cs, cf)
    il, acc, rh = evaluate()
    rows.append(dict(path=path, coord=coord, c_sym=cs, c_flux=cf,
                     variant="frozen", induction_loss=il, induction_acc=acc,
                     random_half_loss=rh, calib_ce_first=None,
                     calib_ce_last=None))
    print(f"[xi-run] {path:8s} t={coord:.1f} (c_sym={cs:.1f}, c_flux={cf:.1f})"
          f" frozen | induction_loss {il:8.4f} acc {acc:.4f}"
          f" | random_half {rh:8.4f}", flush=True)
    if calib_steps > 0:
        c0, c1 = calibrate(calib_steps, calib_lr)
        il2, acc2, rh2 = evaluate()
        rows.append(dict(path=path, coord=coord, c_sym=cs, c_flux=cf,
                         variant="calib", induction_loss=il2,
                         induction_acc=acc2, random_half_loss=rh2,
                         calib_ce_first=c0, calib_ce_last=c1))
        print(f"[xi-run] {path:8s} t={coord:.1f} (c_sym={cs:.1f}, "
              f"c_flux={cf:.1f}) calib  | induction_loss {il2:8.4f} "
              f"acc {acc2:.4f} | random_half {rh2:8.4f} "
              f"| calib CE {c0:.3f}->{c1:.3f}", flush=True)
        restore_scalars()
G.XI_MIX = None
G.XI_TEMP = None

result = {
    "arm": cfg["arm"], "origin": cfg["origin"], "checkpoint_step": cfg["step"],
    "operator": cfg["attn_overrides"], "batch": B, "seqlen": T,
    "seed": cfg["seed"], "paired_batch": True,
    "baseline_xi1": {"induction_loss": base[0], "induction_acc": base[1],
                     "random_half_loss": base[2]},
    "scan": rows,
}
json.dump(result, open(cfg["out_json"], "w"), indent=2)
with open(cfg["out_csv"], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("[xi-run] wrote", cfg["out_json"], "and", cfg["out_csv"], flush=True)
'''


@app.function(
    image=image,
    gpu="H100",
    cpu=16,
    memory=65536,
    timeout=2 * 60 * 60,
    scaledown_window=5,
    volumes={CACHE_DIR: ckpt_vol},
    secrets=[hf_secret],
)
def run_xi_scan(
    arm: str = "amap",
    step: int = -1,
    batch: int = 8,
    seqlen: int = 512,
    seed: int = 1234,
    calib_steps: int = 40,     # 0 disables the calibration rung
    calib_lr: float = 0.01,
    calib_batch: int = 4,
    calib_seed: int = 4321,
    push_repo: str = "sparsetrace/d32ft",
) -> dict:
    assert arm in ARMS, f"arm must be one of {sorted(ARMS)}"
    assert seqlen % 2 == 0
    preset = ARMS[arm]

    os.chdir(REPO_DIR)
    token = os.environ.get("HF_TOKEN", "")
    from huggingface_hub import HfApi
    api = HfApi(token=token) if token else None
    ckpt_vol.reload()

    model_path, meta_path, used_step, origin = _resolve_checkpoint(
        api, token, preset["repo"], preset["run_tag"], step
    )
    _install_xi_gpt()

    label = f"{arm}-step{used_step}"
    workdir = Path("/tmp/xi_sim")
    workdir.mkdir(parents=True, exist_ok=True)
    out_json = workdir / f"xi_scan_{label}.json"
    out_csv = workdir / f"xi_scan_{label}.csv"
    job = {
        "arm": arm, "origin": origin, "step": used_step,
        "vintage": preset["run_tag"] is None,
        "d32_config": D32_CONFIG,
        "model_path": str(model_path), "meta_path": str(meta_path),
        "attn_overrides": {
            "hmap_alpha": preset["hmap_alpha"],
            "hmap_beta": preset["hmap_beta"],
        },
        "batch": batch, "seqlen": seqlen, "seed": seed,
        "calib_steps": calib_steps, "calib_lr": calib_lr,
        "calib_batch": calib_batch, "calib_seed": calib_seed,
        "out_json": str(out_json), "out_csv": str(out_csv),
    }
    (workdir / "job.json").write_text(json.dumps(job))
    (workdir / "probe_run.py").write_text(PROBE_PY)

    _run_streamed(
        f"cd {REPO_DIR} && PYTHONPATH={REPO_DIR} "
        f"{VENV}/bin/python -u {workdir}/probe_run.py {workdir}/job.json"
    )

    if api is not None:
        for p in (out_json, out_csv):
            api.upload_file(path_or_fileobj=str(p),
                            path_in_repo=f"probes/{p.name}", repo_id=push_repo)
        print(f"[xi] archived -> {push_repo}/probes/")
    print("[xi] all done, returning.")
    return json.loads(out_json.read_text())


@app.local_entrypoint()
def main(
    arm: str = "amap",
    step: int = -1,
    batch: int = 8,
    seqlen: int = 512,
    seed: int = 1234,
    calib_steps: int = 40,
    calib_lr: float = 0.01,
    calib_batch: int = 4,
    calib_seed: int = 4321,
    push_repo: str = "sparsetrace/d32ft",
):
    result = run_xi_scan.remote(
        arm=arm, step=step, batch=batch, seqlen=seqlen, seed=seed,
        calib_steps=calib_steps, calib_lr=calib_lr,
        calib_batch=calib_batch, calib_seed=calib_seed,
        push_repo=push_repo,
    )
    scan = result.pop("scan")
    print(json.dumps(result, indent=2))
    print("  path     |  t  | (c_sym, c_flux) | variant | induction_loss  acc     random_half")
    for r in scan:
        print(f" {r['path']:8s} | {r['coord']:.1f} | ({r['c_sym']:.1f}, {r['c_flux']:.1f})       "
              f"| {r['variant']:6s} | {r['induction_loss']:12.4f} "
              f"{r['induction_acc']:.4f}  {r['random_half_loss']:10.4f}")
