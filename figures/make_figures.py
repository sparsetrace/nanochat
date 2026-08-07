"""
figures/make_figures.py — collect results from the experiment HF repos and
produce all four LM-section figures plus reconstructed (revision-merged) data.

Key trick: our harnesses OVERWRITE logs/CSVs on each launch's upload, but HF
repos are git repos — every historical version survives. For each artifact we
walk the repo's commit history, download every distinct version, and merge
rows by step. This reconstructs the full trajectories (e.g. d32ft steps
0->3000 across three launches) with no manual surgery.

Outputs (into figures/):
  fig_conversion.pdf/.png       — recovery curve (+ dashed original-bpb line
                                  if the d32ft-std probe has run)
  fig_samples.pdf/.png          — generation panel (skipped until the two
                                  sampler runs exist)
  fig_scratch_curves.pdf/.png   — d12 three-arm curves (+ baseline replicas
                                  from revision history)
  fig_core_fingerprint.pdf/.png — 22-task paired CORE comparison
  data/*.csv                    — the merged, reconstructed trajectories

Tolerant by design: missing artifacts produce a WARN and a skipped figure,
never a crash — rerun after the pending launches land and the TODO figures
fill in. Requires HF_TOKEN for the private repos.
"""

import csv
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from huggingface_hub import HfApi, hf_hub_download

TOKEN = os.environ.get("HF_TOKEN") or None
API = HfApi(token=TOKEN)

D32FT = "sparsetrace/d32ft"
D32STD = "sparsetrace/d32ft-std"
MAIN = "sparsetrace/nanochat"

OUT = Path(__file__).resolve().parent
DATA = OUT / "data"
DATA.mkdir(parents=True, exist_ok=True)

VAL_RE = re.compile(r"Step (\d+) \| Validation bpb: ([\d.]+)")
STEP_RE = re.compile(r"step (\d+)/(\d+) .*? loss: ([\d.]+) \| lrm: ([\d.]+)")


def warn(msg):
    print(f"[figures] WARN: {msg}", flush=True)


def info(msg):
    print(f"[figures] {msg}", flush=True)


def file_versions(repo_id: str, path_in_repo: str) -> list[str]:
    """All distinct historical versions of a repo file (newest first)."""
    try:
        commits = API.list_repo_commits(repo_id=repo_id)
    except Exception as e:
        warn(f"cannot list commits of {repo_id}: {e}")
        return []
    seen, versions = set(), []
    for c in commits:
        try:
            local = hf_hub_download(
                repo_id=repo_id, filename=path_in_repo,
                revision=c.commit_id, token=TOKEN)
        except Exception:
            continue  # file absent at this revision
        text = Path(local).read_text()
        h = hash(text)
        if h not in seen:
            seen.add(h)
            versions.append(text)
    info(f"{repo_id}/{path_in_repo}: {len(versions)} distinct version(s) "
         f"across history")
    return versions


def merged_val_points(repo_id: str, log_path: str) -> dict:
    """step -> val bpb, merged across all historical train.log versions."""
    points = {}
    for text in file_versions(repo_id, log_path):
        for m in VAL_RE.finditer(text):
            points.setdefault(int(m.group(1)), float(m.group(2)))
    return dict(sorted(points.items()))


def merged_train_rows(repo_id: str, log_path: str) -> dict:
    """step -> train loss, merged across historical train.log versions."""
    rows = {}
    for text in file_versions(repo_id, log_path):
        for m in STEP_RE.finditer(text):
            rows.setdefault(int(m.group(1)), float(m.group(3)))
    return dict(sorted(rows.items()))


def latest_json(repo_id: str, path_in_repo: str):
    try:
        local = hf_hub_download(repo_id=repo_id, filename=path_in_repo,
                                token=TOKEN)
        return json.loads(Path(local).read_text())
    except Exception as e:
        warn(f"missing {repo_id}/{path_in_repo}: {e}")
        return None


def latest_text(repo_id: str, path_in_repo: str):
    try:
        local = hf_hub_download(repo_id=repo_id, filename=path_in_repo,
                                token=TOKEN)
        return Path(local).read_text()
    except Exception as e:
        warn(f"missing {repo_id}/{path_in_repo}: {e}")
        return None


def save_csv(name: str, rows: dict, header: str):
    p = DATA / name
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", header])
        for s, v in rows.items():
            w.writerow([s, v])
    info(f"wrote {p} ({len(rows)} rows)")


def savefig(fig, stem: str):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    info(f"wrote {stem}.pdf/.png")


# ── Figure 1: conversion recovery curve ─────────────────────────────────────
def fig_conversion():
    val = merged_val_points(D32FT, "logs/d32ft-amap_train.log")
    train = merged_train_rows(D32FT, "logs/d32ft-amap_train.log")
    if not val:
        warn("no d32ft val points — skipping fig_conversion")
        return
    save_csv("d32ft_val_bpb.csv", val, "val_bpb")
    save_csv("d32ft_train_loss.csv", train, "train_loss")

    # Original-model anchor from the probe (optional until it runs).
    probe_bpb = None
    probe_log = latest_text(D32STD, "logs/d32-std-probe_train.log")
    if probe_log:
        m = VAL_RE.search(probe_log)
        if m:
            probe_bpb = float(m.group(2))
            info(f"probe original-model bpb: {probe_bpb}")

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    if train:
        ax2 = ax.twinx()
        ts = list(train)
        ax2.plot(ts, [train[t] for t in ts], lw=0.5, alpha=0.25,
                 color="gray", zorder=1)
        ax2.set_ylabel("train loss (nats)", color="gray", fontsize=8)
        ax2.tick_params(labelsize=7, colors="gray")
    ss = list(val)
    ax.plot(ss, [val[s] for s in ss], "o-", ms=3.5, lw=1.4,
            color="tab:blue", zorder=3, label="converted (AMAP)")
    ax.annotate("operator swap", xy=(ss[0], val[ss[0]]),
                xytext=(ss[0] + 250, val[ss[0]] - 0.12),
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.axvline(2250, color="k", lw=0.6, alpha=0.35)
    ax.text(2250, ax.get_ylim()[1], " LR anneal", fontsize=7,
            va="top", alpha=0.7)
    if probe_bpb is not None:
        ax.axhline(probe_bpb, ls="--", color="tab:red", lw=1.1,
                   label=f"original model ({probe_bpb:.3f})")
    ax.set_xlabel("reconditioning step")
    ax.set_ylabel("validation bits per byte")
    ax.legend(fontsize=8, loc="upper right")
    savefig(fig, "fig_conversion")


# ── Figure 2: generation panel ──────────────────────────────────────────────
def fig_samples():
    orig = latest_text(D32FT, "samples/original_T0.8_samples.txt")
    conv = latest_text(D32FT, "samples/d32ft-amap-step3000_T0.8_samples.txt")
    if not (orig and conv):
        warn("sampler outputs not yet on HF — skipping fig_samples "
             "(rerun after the two sample-d32 launches)")
        return

    def first_per_prompt(text, n=3):
        lines = [ln for ln in text.splitlines() if ln and
                 not ln.startswith("#")]
        return lines[:n]

    left, right = first_per_prompt(orig), first_per_prompt(conv)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    for ax, title, lines in [(axes[0], "source (standard attention)", left),
                             (axes[1], "converted (AMAP)", right)]:
        ax.axis("off")
        ax.set_title(title, fontsize=9)
        body = "\n\n".join(
            ln if len(ln) < 340 else ln[:340] + " ..." for ln in lines)
        ax.text(0, 1, body, fontsize=6.2, family="monospace",
                va="top", wrap=True, transform=ax.transAxes)
    savefig(fig, "fig_samples")


# ── Figure 3: from-scratch three-arm curves ─────────────────────────────────
def fig_scratch_curves():
    arms = [("standard", MAIN, "attention/logs/d12-base_train.log",
             "tab:gray"),
            ("AMAP", MAIN, "AMAP/logs/d12-a00_train.log", "tab:blue"),
            ("DMAP", MAIN, "DMAP/logs/d12-a10_train.log", "tab:red")]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    plotted = 0
    for name, repo, log_path, color in arms:
        versions = file_versions(repo, log_path)
        if not versions:
            warn(f"no versions for {name} ({log_path})")
            continue
        # Distinct historical versions of the same log = distinct runs.
        # Plot up to two for the baseline (noise-floor visual), newest one
        # for the variants.
        n_plot = 2 if name == "standard" else 1
        for k, text in enumerate(versions[:n_plot]):
            points = {int(m.group(1)): float(m.group(2))
                      for m in VAL_RE.finditer(text)}
            if not points:
                continue
            ss = sorted(points)
            label = name if k == 0 else f"{name} (replica)"
            ax.plot(ss, [points[s] for s in ss], "-", lw=1.3, color=color,
                    alpha=1.0 if k == 0 else 0.55, label=label)
            save_csv(f"d12_{name}{'' if k == 0 else '_replica'}_val_bpb.csv",
                     dict(sorted(points.items())), "val_bpb")
            plotted += 1
    if plotted == 0:
        warn("no from-scratch curves found — skipping fig_scratch_curves")
        plt.close(fig)
        return
    ax.set_xlabel("training step")
    ax.set_ylabel("validation bits per byte")
    ax.set_ylim(0.82, 1.05)
    ax.legend(fontsize=8)
    savefig(fig, "fig_scratch_curves")


# ── Figure 4: CORE per-task fingerprint ─────────────────────────────────────
def fig_core_fingerprint():
    orig = latest_json(D32FT, "evals/original_core.json")
    conv = latest_json(D32FT, "evals/d32ft-amap-step3000_core.json")
    if not (orig and conv):
        warn("CORE jsons incomplete — skipping fig_core_fingerprint")
        return
    o, c = orig["centered_results"], conv["centered_results"]
    tasks = sorted(set(o) & set(c), key=lambda t: c[t] - o[t])
    diffs = [c[t] - o[t] for t in tasks]
    with open(DATA / "core_fingerprint.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "original", "converted", "diff"])
        for t in tasks:
            w.writerow([t, o[t], c[t], c[t] - o[t]])

    fig, ax = plt.subplots(figsize=(5.2, 5.6))
    y = range(len(tasks))
    ax.barh(list(y), diffs, color=["tab:red" if d < 0 else "tab:blue"
                                   for d in diffs], alpha=0.85)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(list(y))
    ax.set_yticklabels([t.replace("bigbench_", "bb_") for t in tasks],
                       fontsize=7)
    ax.set_xlabel("centered accuracy: converted $-$ original")
    ax.set_title(
        f"CORE: original {orig['core_metric']:.3f} -> "
        f"converted {conv['core_metric']:.3f} (GPT-2: 0.257)", fontsize=9)
    savefig(fig, "fig_core_fingerprint")


if __name__ == "__main__":
    info("collecting artifacts and building figures...")
    fig_conversion()
    fig_samples()
    fig_scratch_curves()
    fig_core_fingerprint()
    info("done. (WARN lines above list anything skipped — rerun after "
         "pending launches to fill in.)")
