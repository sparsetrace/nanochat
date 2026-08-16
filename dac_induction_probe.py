"""
DAC induction-current probe for nanochat d32.

Outer mode (Modal harness):
    modal run dac_induction_probe.py --arm all
    modal run dac_induction_probe.py --arm amap --lengths 64,128,256,512

Runtime mode is launched automatically inside the repo venv. It reproduces nanochat.icl_eval.induction_score exactly at the level of
synthetic examples and reported metrics, but microbatches the model forwards so
longer L values do not materialize a batch_size x (2L-1) x vocab fp32 logit
tensor all at once. Forward hooks on c_q/c_k measure the antisymmetric DAC
score field on the induction current support.

For repeated-random [s ; s] sequences of half-length L, the existing nanochat
benchmark scores query rows t=L,...,2L-2. The induction source is

    j_ind(t) = t - L + 1,

so the oriented induction current is supported on t -> j_ind(t).

For each layer/head, using the exact post-RoPE, post-QK-norm, post-1.2x q,k
vectors scored by nanochat,

    A_ij = 1/(2 sqrt(d_h)) [ <q_i,k_j> - <k_i,q_j> ].

This is the exact flux sector used by AMAP/HMAP. For standard attention it is
the antisymmetric sector of the ordinary qk^T logit matrix. For HMAP the
active flux is (1-alpha) A, so CMAP/DMAP have active flux zero even though the
latent q,k-defined A can still be measured.

The probe also measures a small local offset profile j_ind + delta. This gives
a cheap control for generic long-range directionality: a task-specific
induction head should peak near delta=0 rather than simply showing a broad
long-range antisymmetric bias.
"""

from __future__ import annotations

import sys

_RUNTIME = "--runtime" in sys.argv


if _RUNTIME:
    # ---------------------------------------------------------------------
    # Inner runtime: executed with /root/nanochat/.venv/bin/python, where
    # torch and the nanochat package live. Avoid importing Modal here.
    # ---------------------------------------------------------------------
    import argparse
    import gc
    import json
    import math
    from pathlib import Path

    import torch
    import torch.nn.functional as F

    from nanochat.common import COMPUTE_DTYPE
    from nanochat.gpt import GPT, GPTConfig, apply_rotary_emb, norm
    from nanochat.tokenizer import get_tokenizer


    def _parse_csv_ints(text: str) -> list[int]:
        vals = [int(x.strip()) for x in text.split(",") if x.strip()]
        if not vals:
            raise ValueError("expected at least one integer")
        return vals


    @torch.no_grad()
    def _induction_score_microbatched(
        model, vocab_size, device, *, seq_len=256, logical_batch_size=16,
        num_batches=4, seed=1234, micro_batch_size=0,
    ):
        """Memory-safe replica of nanochat.icl_eval.induction_score.

        The logical synthetic batches are generated with the same CPU generator,
        seed, shapes, and ordering as nanochat's benchmark. Only the model forward
        is split into smaller microbatches. Because every example contributes the
        same number of scored positions, the final means are identical to averaging
        the original logical batches (up to harmless kernel-level roundoff from a
        different physical batch size).
        """
        L = int(seq_len)
        B = int(logical_batch_size)
        mb = B if micro_batch_size <= 0 else min(B, int(micro_batch_size))
        g = torch.Generator().manual_seed(seed)

        random_loss_sum = 0.0
        random_count = 0
        induction_loss_sum = 0.0
        induction_count = 0
        induction_correct = 0

        for _ in range(num_batches):
            # Generate the full logical batch in one call, exactly as the checked-in
            # nanochat benchmark does. Microbatch only AFTER sampling so the eval set
            # is unchanged.
            s_cpu = torch.randint(0, vocab_size, (B, L), generator=g)
            for b0 in range(0, B, mb):
                s = s_cpu[b0:b0 + mb].to(device)
                seq = torch.cat([s, s], dim=1)
                x, y = seq[:, :-1], seq[:, 1:]
                logits = model(x)
                losses = F.cross_entropy(
                    logits.transpose(1, 2).float(), y, reduction="none"
                )
                preds = logits.argmax(dim=-1)

                r = losses[:, :L - 1]
                z = losses[:, L:]
                random_loss_sum += r.double().sum().item()
                random_count += r.numel()
                induction_loss_sum += z.double().sum().item()
                induction_count += z.numel()
                induction_correct += (preds[:, L:] == y[:, L:]).sum().item()

                del s, seq, x, y, logits, losses, preds, r, z

        return {
            "random_half_loss": random_loss_sum / max(random_count, 1),
            "induction_loss": induction_loss_sum / max(induction_count, 1),
            "induction_acc": induction_correct / max(induction_count, 1),
            "logical_batch_size": B,
            "micro_batch_size": mb,
        }


    def _auto_micro_batch_size(logical_batch_size: int, L: int, max_forward_tokens: int) -> int:
        """Keep B_micro * T near a known-safe activation/logit budget.

        GPT.forward materializes fp32 vocabulary logits of shape (B,T,V), so its
        dominant memory term scales approximately with B*T. The original probe was
        observed safe at B=16, L=256 (T=511), hence the default budget of 8192
        sequence-tokens per forward.
        """
        T = 2 * int(L) - 1
        return max(1, min(int(logical_batch_size), int(max_forward_tokens) // T))


    def _build_d32_model(
        *,
        vocab_size: int,
        depth: int,
        context_len: int,
        attn_variant: str,
        hmap_alpha: float,
        hmap_beta: float,
        device: torch.device,
    ) -> GPT:
        # Mirrors scripts/base_train.py's d32 construction exactly.
        aspect_ratio = 64
        head_dim = 128
        base_dim = depth * aspect_ratio
        model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
        num_heads = model_dim // head_dim
        config = GPTConfig(
            sequence_len=context_len,
            vocab_size=vocab_size,
            n_layer=depth,
            n_head=num_heads,
            n_kv_head=num_heads,
            n_embd=model_dim,
            window_pattern="L",
            attn_variant=attn_variant,
            hmap_alpha=hmap_alpha,
            hmap_beta=hmap_beta,
            witten=False,
            bidirectional=False,
        )
        with torch.device("meta"):
            model = GPT(config)
        model.to_empty(device=device)
        model.init_weights()
        return model


    def _load_weights_vintage_compatible(model: GPT, model_path: str) -> list[str]:
        """Same strict-with-allowlist + graft-neutral fill as icl_probe.py."""
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        own = model.state_dict()
        missing = sorted(set(own) - set(state))
        unexpected = sorted(set(state) - set(own))
        shape_bad = [
            k for k in set(own) & set(state)
            if tuple(state[k].shape) != tuple(own[k].shape)
        ]
        if shape_bad:
            raise RuntimeError(f"shape mismatch: {shape_bad[:8]}")
        if unexpected:
            raise RuntimeError(f"unexpected checkpoint keys: {unexpected[:8]}")
        allow = (
            "value_embed", "ve_gate", "resid_lambda", "x0_lambda",
            "smear_gate", "smear_lambda", "backout_lambda",
        )
        bad = [k for k in missing if not any(a in k for a in allow)]
        if bad:
            raise RuntimeError(f"missing keys outside vintage allowlist: {bad[:8]}")

        model.load_state_dict(state, strict=False)
        del state, own
        gc.collect()

        # Match the neutral fill used by the pinned ICL probe harness.
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name not in missing:
                    continue
                if "value_embeds" in name:
                    p.zero_()
                elif "resid_lambda" in name:
                    p.fill_(1.0)
                elif "x0_lambda" in name or "backout_lambda" in name:
                    p.zero_()
        return missing


    class DACInductionCollector:
        """Layer/head resolved A·J measurements via c_q/c_k forward hooks."""

        def __init__(self, model: GPT, offsets: list[int], active_flux_coeff: float):
            self.model = model
            self.offsets = offsets
            self.active_flux_coeff = float(active_flux_coeff)
            self.n_layer = model.config.n_layer
            self.n_head = model.config.n_head
            self.head_dim = model.config.n_embd // model.config.n_head
            if model.config.n_kv_head != model.config.n_head:
                raise RuntimeError(
                    "DAC A extraction currently requires n_kv_head == n_head; "
                    "the d32 checkpoints satisfy this."
                )

            self._handles = []
            self._pending_q: dict[int, torch.Tensor] = {}
            self.current_L: int | None = None
            self._reset_stats()

            for layer_idx, block in enumerate(model.transformer.h):
                self._handles.append(
                    block.attn.c_q.register_forward_hook(self._make_q_hook(layer_idx))
                )
                self._handles.append(
                    block.attn.c_k.register_forward_hook(self._make_k_hook(layer_idx))
                )

        def _reset_stats(self):
            nl, nh, no = self.n_layer, self.n_head, len(self.offsets)
            self.sum = torch.zeros(nl, nh, dtype=torch.float64)
            self.abs_sum = torch.zeros(nl, nh, dtype=torch.float64)
            self.sq_sum = torch.zeros(nl, nh, dtype=torch.float64)
            self.count = torch.zeros(nl, dtype=torch.float64)
            self.offset_sum = torch.zeros(nl, nh, no, dtype=torch.float64)
            self.offset_abs_sum = torch.zeros(nl, nh, no, dtype=torch.float64)
            self.offset_count = torch.zeros(nl, no, dtype=torch.float64)
            self._pending_q.clear()

        def begin(self, L: int):
            self.current_L = int(L)
            self._reset_stats()

        def _make_q_hook(self, layer_idx: int):
            def hook(_module, _inputs, output):
                self._pending_q[layer_idx] = output
            return hook

        @staticmethod
        def _pair_A(qh, kh, t_idx, j_idx, scale: float):
            # qh, kh: (B,H,T,D). The multiply is bf16/fp16 as in the model;
            # reduction accumulates in fp32 for a stable measurement.
            qt = qh[:, :, t_idx, :]
            kj = kh[:, :, j_idx, :]
            kt = kh[:, :, t_idx, :]
            qj = qh[:, :, j_idx, :]
            qk = torch.sum(qt * kj, dim=-1, dtype=torch.float32)
            kq = torch.sum(kt * qj, dim=-1, dtype=torch.float32)
            return 0.5 * (qk - kq) * scale  # (B,H,N_edges)

        def _make_k_hook(self, layer_idx: int):
            def hook(_module, _inputs, k_raw):
                if self.current_L is None:
                    return
                q_raw = self._pending_q.pop(layer_idx, None)
                if q_raw is None:
                    raise RuntimeError(f"layer {layer_idx}: c_k hook fired without c_q output")

                B, T, _ = q_raw.shape
                L = self.current_L
                expected_T = 2 * L - 1
                if T != expected_T:
                    raise RuntimeError(
                        f"layer {layer_idx}: expected induction input T={expected_T} for L={L}, got {T}"
                    )

                H, D = self.n_head, self.head_dim
                q = q_raw.view(B, T, H, D)
                k = k_raw.view(B, T, H, D)
                cos = self.model.cos[:, :T]
                sin = self.model.sin[:, :T]

                # Reconstruct exactly the q,k vectors the attention operator sees.
                q = apply_rotary_emb(q, cos, sin)
                k = apply_rotary_emb(k, cos, sin)
                q = norm(q) * 1.2
                k = norm(k) * 1.2
                qh = q.transpose(1, 2)
                kh = k.transpose(1, 2)
                scale = D ** -0.5

                # Existing nanochat induction benchmark:
                # t = L,...,2L-2 and j_ind = t-L+1 = 1,...,L-1.
                t = torch.arange(L, 2 * L - 1, device=q.device)
                j = t - L + 1
                a = self._pair_A(qh, kh, t, j, scale)

                self.sum[layer_idx] += a.sum(dim=(0, 2)).double().cpu()
                self.abs_sum[layer_idx] += a.abs().sum(dim=(0, 2)).double().cpu()
                self.sq_sum[layer_idx] += a.square().sum(dim=(0, 2)).double().cpu()
                self.count[layer_idx] += B * a.shape[-1]

                # Local positional control profile around the true source.
                # Restrict controls to the first copy 0,...,L-1.
                for oi, delta in enumerate(self.offsets):
                    jj = j + delta
                    valid = (jj >= 0) & (jj < L)
                    if not bool(valid.any()):
                        continue
                    aa = self._pair_A(qh, kh, t[valid], jj[valid], scale)
                    self.offset_sum[layer_idx, :, oi] += aa.sum(dim=(0, 2)).double().cpu()
                    self.offset_abs_sum[layer_idx, :, oi] += aa.abs().sum(dim=(0, 2)).double().cpu()
                    self.offset_count[layer_idx, oi] += B * aa.shape[-1]

                del q_raw, q, k, qh, kh, a
            return hook

        def finish(self) -> dict:
            if self._pending_q:
                raise RuntimeError(f"unmatched q hooks remain for layers {sorted(self._pending_q)}")

            count = self.count.clamp_min(1.0).unsqueeze(1)
            raw_mean = self.sum / count
            raw_abs_mean = self.abs_sum / count
            raw_rms = torch.sqrt(self.sq_sum / count)

            oc = self.offset_count.clamp_min(1.0).unsqueeze(1)
            offset_mean = self.offset_sum / oc
            offset_abs_mean = self.offset_abs_sum / oc

            # Local contrast: true induction edge minus immediate neighbors.
            local_contrast = torch.full_like(raw_mean, float("nan"))
            if 0 in self.offsets:
                i0 = self.offsets.index(0)
                neigh = []
                for d in (-1, 1):
                    if d in self.offsets:
                        neigh.append(offset_mean[:, :, self.offsets.index(d)])
                if neigh:
                    neighbor_mean = torch.stack(neigh, dim=0).mean(dim=0)
                    local_contrast = offset_mean[:, :, i0] - neighbor_mean

            c = self.active_flux_coeff
            return {
                "definition": "A_ij = 0.5*(q_i.k_j - k_i.q_j)/sqrt(head_dim)",
                "orientation": "query row i -> key column j; measured pre-causal-mask as the underlying score kernel",
                "active_flux_coefficient": c,
                "raw_mean": raw_mean.tolist(),
                "raw_abs_mean": raw_abs_mean.tolist(),
                "raw_rms": raw_rms.tolist(),
                "active_mean": (c * raw_mean).tolist(),
                "active_abs_mean": (abs(c) * raw_abs_mean).tolist(),
                "local_contrast_raw": local_contrast.tolist(),
                "local_contrast_active": (c * local_contrast).tolist(),
                "offsets": self.offsets,
                "offset_raw_mean": offset_mean.tolist(),
                "offset_raw_abs_mean": offset_abs_mean.tolist(),
                "edge_count_per_layer": self.count.tolist(),
                "offset_edge_count_per_layer": self.offset_count.tolist(),
            }

        def close(self):
            for h in self._handles:
                h.remove()
            self._handles.clear()


    def _top_heads(stats: dict, k: int = 8) -> list[dict]:
        x = torch.tensor(stats["local_contrast_raw"], dtype=torch.float32)
        if torch.isnan(x).all():
            x = torch.tensor(stats["raw_mean"], dtype=torch.float32)
        score = torch.nan_to_num(x.abs(), nan=-1.0)
        n = min(k, score.numel())
        vals, idx = torch.topk(score.flatten(), n)
        H = score.shape[1]
        raw = torch.tensor(stats["raw_mean"], dtype=torch.float32)
        con = torch.tensor(stats["local_contrast_raw"], dtype=torch.float32)
        out = []
        for v, flat in zip(vals.tolist(), idx.tolist()):
            layer, head = divmod(flat, H)
            out.append({
                "layer": layer,
                "head": head,
                "abs_rank_score": v,
                "raw_mean": float(raw[layer, head]),
                "local_contrast_raw": float(con[layer, head]),
            })
        return out


    def runtime_main():
        p = argparse.ArgumentParser()
        p.add_argument("--model-path", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--arm", required=True)
        p.add_argument("--origin", default="")
        p.add_argument("--checkpoint-step", type=int, required=True)
        p.add_argument("--attn-variant", choices=["standard", "hmap"], required=True)
        p.add_argument("--hmap-alpha", type=float, default=0.0)
        p.add_argument("--hmap-beta", type=float, default=0.0)
        p.add_argument("--depth", type=int, default=32)
        p.add_argument("--context-len", type=int, default=2048)
        p.add_argument("--lengths", default="32,64,128,256,512")
        p.add_argument("--offsets", default="-2,-1,0,1,2")
        p.add_argument("--batch-size", type=int, default=16,
                       help="logical synthetic batch size; 16 matches nanochat")
        p.add_argument("--micro-batch-size", type=int, default=0,
                       help="physical forward batch (0 = auto from max-forward-tokens)")
        p.add_argument("--max-forward-tokens", type=int, default=8192,
                       help="auto microbatch budget: micro_batch*(2L-1) <= this")
        p.add_argument("--num-batches", type=int, default=4)
        p.add_argument("--seed", type=int, default=1234)
        args = p.parse_args()

        if not torch.cuda.is_available():
            raise RuntimeError("DAC induction probe expects a CUDA GPU")
        device = torch.device("cuda")
        lengths = _parse_csv_ints(args.lengths)
        offsets = _parse_csv_ints(args.offsets)
        for L in lengths:
            if 2 * L - 1 > args.context_len:
                raise ValueError(
                    f"L={L} gives input length {2*L-1}, exceeding context {args.context_len}"
                )

        tokenizer = get_tokenizer()
        vocab_size = tokenizer.get_vocab_size()
        model = _build_d32_model(
            vocab_size=vocab_size,
            depth=args.depth,
            context_len=args.context_len,
            attn_variant=args.attn_variant,
            hmap_alpha=args.hmap_alpha,
            hmap_beta=args.hmap_beta,
            device=device,
        )
        missing = _load_weights_vintage_compatible(model, args.model_path)
        model.eval()

        active_coeff = 1.0 if args.attn_variant == "standard" else (1.0 - args.hmap_alpha)
        collector = DACInductionCollector(model, offsets, active_coeff)

        result = {
            "probe": "dac_induction_current",
            "arm": args.arm,
            "origin": args.origin,
            "checkpoint_step": args.checkpoint_step,
            "operator": {
                "attn_variant": args.attn_variant,
                "hmap_alpha": args.hmap_alpha,
                "hmap_beta": args.hmap_beta,
                "active_flux_coefficient": active_coeff,
            },
            "model": {
                "depth": model.config.n_layer,
                "n_head": model.config.n_head,
                "head_dim": model.config.n_embd // model.config.n_head,
                "n_embd": model.config.n_embd,
                "context_len": args.context_len,
                "compute_dtype": str(COMPUTE_DTYPE),
                "vocab_size": vocab_size,
                "vintage_missing_keys": missing,
            },
            "benchmark": {
                "generator": "nanochat.icl_eval.induction_score construction + metrics, memory-safe microbatched forwards",
                "sequence": "[s ; s], s ~ Uniform(vocab)^L",
                "current_support": "t -> t-L+1 for t=L,...,2L-2",
                "seed": args.seed,
                "logical_batch_size": args.batch_size,
                "requested_micro_batch_size": args.micro_batch_size,
                "max_forward_tokens": args.max_forward_tokens,
                "num_batches": args.num_batches,
                "lengths": lengths,
                "offsets": offsets,
            },
            "by_length": {},
        }

        try:
            for L in lengths:
                if args.micro_batch_size > 0:
                    micro_batch = min(args.batch_size, args.micro_batch_size)
                else:
                    micro_batch = _auto_micro_batch_size(
                        args.batch_size, L, args.max_forward_tokens
                    )

                # If a particular CUDA/kernel configuration still exceeds memory,
                # restart this L from the same seed with half the physical batch.
                # collector.begin() discards any partial A statistics from the failed
                # attempt, so retries cannot double-count edges.
                while True:
                    collector.begin(L)
                    try:
                        with torch.inference_mode():
                            behavior = _induction_score_microbatched(
                                model,
                                vocab_size,
                                device,
                                seq_len=L,
                                logical_batch_size=args.batch_size,
                                num_batches=args.num_batches,
                                seed=args.seed,
                                micro_batch_size=micro_batch,
                            )
                        stats = collector.finish()
                        break
                    except torch.OutOfMemoryError:
                        if micro_batch <= 1:
                            raise
                        old_mb = micro_batch
                        micro_batch = max(1, micro_batch // 2)
                        print(
                            f"[dac] CUDA OOM at arm={args.arm} L={L} "
                            f"micro_batch={old_mb}; retrying from scratch with "
                            f"micro_batch={micro_batch}",
                            flush=True,
                        )
                        gc.collect()
                        torch.cuda.empty_cache()

                tops = _top_heads(stats)
                result["by_length"][str(L)] = {
                    "behavior": behavior,
                    "dac": stats,
                    "top_heads": tops,
                }
                top_txt = ", ".join(
                    f"L{h['layer']}:H{h['head']}={h['local_contrast_raw']:+.4f}"
                    for h in tops[:5]
                )
                print(
                    f"[dac] arm={args.arm} L={L} "
                    f"loss={behavior['induction_loss']:.4f} "
                    f"acc={behavior['induction_acc']:.4f} "
                    f"microB={behavior['micro_batch_size']} top={top_txt}",
                    flush=True,
                )
                gc.collect()
                torch.cuda.empty_cache()
        finally:
            collector.close()

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"[dac] wrote {out}", flush=True)


    if __name__ == "__main__":
        # Remove the sentinel so argparse does not see it.
        sys.argv.remove("--runtime")
        runtime_main()

else:
    # ---------------------------------------------------------------------
    # Outer Modal harness. This process intentionally uses system Python;
    # torch exists only in the repo venv created by uv sync --extra gpu.
    # ---------------------------------------------------------------------
    import json
    import os
    import re
    import shutil
    import subprocess
    from pathlib import Path

    import modal

    app = modal.App("nanochat-dac-induction-probe")

    TOKENIZER_REPO = "karpathy/nanochat-d32"
    SOURCE_TOKENIZER = ("tokenizer.pkl", "token_bytes.pt")
    KARPATHY_MODEL = "model_000650.pt"
    KARPATHY_META = "meta_000650.json"

    ARMS = {
        "original": dict(
            repo="karpathy/nanochat-d32", run_tag=None,
            attn_variant="standard", hmap_alpha=0.0, hmap_beta=0.0,
        ),
        "amap": dict(
            repo="sparsetrace/d32ft", run_tag="d32ft-amap",
            attn_variant="hmap", hmap_alpha=0.0, hmap_beta=0.0,
        ),
        "cmap": dict(
            repo="sparsetrace/d32cmap", run_tag="d32cmap",
            attn_variant="hmap", hmap_alpha=1.0, hmap_beta=1.0,
        ),
        "regrafted": dict(
            repo="sparsetrace/amap2nanochat", run_tag="amap2nanochat",
            attn_variant="standard", hmap_alpha=0.0, hmap_beta=0.0,
        ),
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
        .env({
            "PATH": "/root/.cargo/bin:/root/.local/bin:/usr/local/bin:/usr/bin:/bin",
            "OMP_NUM_THREADS": "1",
            "HF_HUB_DISABLE_XET": "1",
            "NANOCHAT_BASE_DIR": CACHE_DIR,
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "TORCHINDUCTOR_CACHE_DIR": f"{CACHE_DIR}/inductor_cache",
            "TRITON_CACHE_DIR": f"{CACHE_DIR}/triton_cache",
        })
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


    def _run_streamed(cmd: list[str], log_path: Path | None = None):
        print("[dac] running:", " ".join(cmd), flush=True)
        log_f = open(log_path, "a") if log_path else None
        try:
            proc = subprocess.Popen(
                cmd,
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
            raise RuntimeError(f"command failed with exit code {proc.returncode}: {' '.join(cmd)}")


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
        print(f"[dac] installed exact Karpathy d32 tokenizer -> {tok_dir}")


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
        if run_tag is None:
            seed_dir = Path(CACHE_DIR) / "bootstrap" / "karpathy-d32"
            model = _hf_download_into(
                repo, KARPATHY_MODEL, seed_dir / KARPATHY_MODEL, token=None
            )
            _hf_download_into(repo, KARPATHY_META, seed_dir / KARPATHY_META, token=None)
            return model, 650, f"{repo}:{KARPATHY_MODEL}"

        ckpt_dir = Path(CACHE_DIR) / "base_checkpoints" / run_tag
        local_names = (
            [p.name for p in ckpt_dir.iterdir() if p.is_file()]
            if ckpt_dir.exists() else []
        )
        local_steps = _weights_steps(local_names, "")
        if local_steps and (step in local_steps or step < 0):
            use = step if step > 0 else local_steps[-1]
            print(f"[dac] checkpoint (volume): {run_tag} step {use}")
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
            model = _hf_download_into(
                repo,
                f"{prefix}model_{use:06d}.pt",
                dest / f"model_{use:06d}.pt",
                token=token,
            )
            # Fetch meta too so the volume/HF cache has complete checkpoint pairs.
            _hf_download_into(
                repo,
                f"{prefix}meta_{use:06d}.json",
                dest / f"meta_{use:06d}.json",
                token=token,
            )
            print(f"[dac] checkpoint (HF): {repo}/{prefix} step {use}")
            return model, use, f"{repo}:{prefix}step-{use}"

        raise RuntimeError(
            f"No checkpoint with model+meta found for {run_tag} "
            f"(step {step}) on Volume or in {repo}"
        )


    @app.function(
        image=image,
        gpu="A10G",
        cpu=16,
        memory=65536,
        timeout=6 * 60 * 60,
        retries=0,
        scaledown_window=5,
        volumes={CACHE_DIR: ckpt_vol},
        secrets=[hf_secret],
    )
    def run_probe(
        arm: str = "all",
        step: int = -1,
        lengths: str = "32,64,128,256,512",
        offsets: str = "-2,-1,0,1,2",
        batch_size: int = 16,
        micro_batch_size: int = 0,
        max_forward_tokens: int = 8192,
        num_batches: int = 4,
        seed: int = 1234,
        push_repo: str = "sparsetrace/d32ft",
    ) -> dict:
        if arm != "all" and arm not in ARMS:
            raise ValueError(f"arm must be 'all' or one of {sorted(ARMS)}")

        os.chdir(REPO_DIR)
        token = os.environ.get("HF_TOKEN", "")
        from huggingface_hub import HfApi
        api = HfApi(token=token or None)

        _ensure_source_tokenizer()
        ckpt_vol.reload()

        logs_dir = Path(CACHE_DIR) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        python = f"{VENV}/bin/python"
        script = f"{REPO_DIR}/dac_induction_probe.py"
        arms = list(ARMS) if arm == "all" else [arm]
        all_results = {}

        length_tag = "L" + "-".join(x.strip() for x in lengths.split(",") if x.strip())

        for arm_name in arms:
            preset = ARMS[arm_name]
            model_path, used_step, origin = _resolve_checkpoint(
                api, token or None, preset["repo"], preset["run_tag"], step
            )
            label = f"dac-induction-{arm_name}-step{used_step}-{length_tag}"
            out_path = logs_dir / f"{label}.json"
            log_path = logs_dir / f"{label}.log"
            out_path.unlink(missing_ok=True)
            log_path.unlink(missing_ok=True)

            cmd = [
                python,
                script,
                "--runtime",
                "--model-path", str(model_path),
                "--output", str(out_path),
                "--arm", arm_name,
                "--origin", origin,
                "--checkpoint-step", str(used_step),
                "--attn-variant", preset["attn_variant"],
                "--hmap-alpha", str(preset["hmap_alpha"]),
                "--hmap-beta", str(preset["hmap_beta"]),
                "--lengths", lengths,
                f"--offsets={offsets}",
                "--batch-size", str(batch_size),
                "--micro-batch-size", str(micro_batch_size),
                "--max-forward-tokens", str(max_forward_tokens),
                "--num-batches", str(num_batches),
                "--seed", str(seed),
            ]
            _run_streamed(cmd, log_path=log_path)
            all_results[arm_name] = json.loads(out_path.read_text())

            if token:
                api.upload_file(
                    path_or_fileobj=str(out_path),
                    path_in_repo=f"probes/dac_induction/{out_path.name}",
                    repo_id=push_repo,
                )
                api.upload_file(
                    path_or_fileobj=str(log_path),
                    path_in_repo=f"probes/dac_induction/{log_path.name}",
                    repo_id=push_repo,
                )
                print(f"[dac] archived {arm_name} -> {push_repo}/probes/dac_induction/")

        bundle = {
            "probe": "dac_induction_current_bundle",
            "requested_arm": arm,
            "lengths": [int(x) for x in lengths.split(",") if x.strip()],
            "offsets": [int(x) for x in offsets.split(",") if x.strip()],
            "logical_batch_size": batch_size,
            "micro_batch_size": micro_batch_size,
            "max_forward_tokens": max_forward_tokens,
            "num_batches": num_batches,
            "seed": seed,
            "arms": all_results,
        }
        bundle_name = f"dac-induction-{arm}-{length_tag}.json"
        bundle_path = logs_dir / bundle_name
        bundle_path.write_text(json.dumps(bundle, indent=2) + "\n")

        if token:
            api.upload_file(
                path_or_fileobj=str(bundle_path),
                path_in_repo=f"probes/dac_induction/{bundle_name}",
                repo_id=push_repo,
            )
            print(f"[dac] archived bundle -> {push_repo}/probes/dac_induction/{bundle_name}")

        ckpt_vol.commit()
        print("[dac] all done")
        return bundle


    @app.local_entrypoint()
    def main(
        arm: str = "all",
        step: int = -1,
        lengths: str = "32,64,128,256,512",
        offsets: str = "-2,-1,0,1,2",
        batch_size: int = 16,
        micro_batch_size: int = 0,
        max_forward_tokens: int = 8192,
        num_batches: int = 4,
        seed: int = 1234,
        push_repo: str = "sparsetrace/d32ft",
    ):
        result = run_probe.remote(
            arm=arm,
            step=step,
            lengths=lengths,
            offsets=offsets,
            batch_size=batch_size,
            micro_batch_size=micro_batch_size,
            max_forward_tokens=max_forward_tokens,
            num_batches=num_batches,
            seed=seed,
            push_repo=push_repo,
        )
        # Keep terminal output compact; the full tensor lives in the JSON artifact.
        summary = {
            name: {
                "checkpoint_step": r["checkpoint_step"],
                "operator": r["operator"],
                "behavior": {
                    L: vals["behavior"] for L, vals in r["by_length"].items()
                },
            }
            for name, r in result["arms"].items()
        }
        print(json.dumps(summary, indent=2))
