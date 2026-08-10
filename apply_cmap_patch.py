#!/usr/bin/env python3
"""
apply_cmap_patch.py — add the kinetic-signature coordinate hmap_beta to
nanochat/gpt.py, completing the (beta, alpha) operator square:

    s(beta, alpha)_ij = 1/2<m_i,m_j> - beta * 1/2<n_i,n_j>
                        + (1-alpha) * 1/2(<q_i,k_j> - <k_i,q_j>)
                        + alpha * 1/2(g_i - g_j)

    m = (q+k)/sqrt(2),  n = (q-k)/sqrt(2),  g_i = <q_i,k_i>

    (0,0) = AMAP      (PSD kinetic + flux)
    (0,1) = DMAP      (PSD kinetic + Doob)
    (1,0) = standard  (indefinite kinetic + flux == <q_i,k_j> exactly, eager)
    (1,1) = CMAP      (indefinite kinetic + Doob)

No parameter changes: n is built from the same c_q/c_k projections, and
g_i = <q_i,k_i> = 1/2||m_i||^2 - 1/2||n_i||^2 is ALREADY the indefinite
symmetric potential (the diagonal of a bilinear form sees only its symmetric
part), so the existing g computation is untouched.

Anchor-based, same philosophy as the d32ft ephemeral trainer: every edit
asserts its anchor appears exactly once, else aborts without writing.

Usage:  python apply_cmap_patch.py [path/to/nanochat/gpt.py]
        (default path: nanochat/gpt.py; writes in place, backup at .pre_cmap)
"""

import sys
from pathlib import Path

EDITS = [
    # ── 1. module docstring: document beta under the alpha homotopy lines ──
    (
        """    alpha = 0  ->  AMAP   (kinetic + flux)        [exact reduction, same branch]
    alpha = 1  ->  DMAP face (kinetic + Doob)
    0 < alpha < 1 -> homotopy between the curl-only and gradient-only faces.
""",
        """    alpha = 0  ->  AMAP   (kinetic + flux)        [exact reduction, same branch]
    alpha = 1  ->  DMAP face (kinetic + Doob)
    0 < alpha < 1 -> homotopy between the curl-only and gradient-only faces.

    beta (hmap_beta) restores the NSD kinetic part -beta/2 <n_i, n_j> with
    n = (q - k)/sqrt(2), i.e. the full indefinite symmetric kernel
    W_S = 1/2 W_M W_M^T - 1/2 W_N W_N^T at beta=1. The square:
    (beta,alpha) = (0,0) AMAP, (0,1) DMAP, (1,0) standard attention
    (reproduced exactly in the eager branch), (1,1) CMAP
    (indefinite kinetic + Doob). g is unchanged: diag(R W R^T) sees only
    the symmetric sector, so g_i = <q_i,k_i> = 1/2||m_i||^2 - 1/2||n_i||^2
    is already the indefinite potential.
""",
    ),
    # ── 2. GPTConfig: hmap_beta field after hmap_alpha ─────────────────────
    (
        """    # HMAP homotopy coordinate: 0 = AMAP (kinetic+flux), 1 = DMAP face (kinetic+Doob).
    hmap_alpha: float = 0.0
""",
        """    # HMAP homotopy coordinate: 0 = AMAP (kinetic+flux), 1 = DMAP face (kinetic+Doob).
    hmap_alpha: float = 0.0
    # Kinetic signature coordinate: 0 = PSD kinetic (1/2 W_M W_M^T only),
    # 1 = full indefinite symmetric kernel W_S = 1/2 W_M W_M^T - 1/2 W_N W_N^T.
    # (beta, alpha): (0,0)=AMAP, (0,1)=DMAP, (1,0)=standard (eager), (1,1)=CMAP.
    hmap_beta: float = 0.0
""",
    ),
    # ── 3. CausalSelfAttention.__init__: mirror the alpha attribute ────────
    (
        """        self.attn_variant = config.attn_variant
        self.hmap_alpha = config.hmap_alpha
""",
        """        self.attn_variant = config.attn_variant
        self.hmap_alpha = config.hmap_alpha
        self.hmap_beta = config.hmap_beta
""",
    ),
    # ── 4. range assert next to the alpha one ──────────────────────────────
    (
        """            assert 0.0 <= self.hmap_alpha <= 1.0, f"hmap_alpha in [0,1], got {self.hmap_alpha}"
""",
        """            assert 0.0 <= self.hmap_alpha <= 1.0, f"hmap_alpha in [0,1], got {self.hmap_alpha}"
            assert 0.0 <= self.hmap_beta <= 1.0, f"hmap_beta in [0,1], got {self.hmap_beta}"
""",
    ),
    # ── 5. forward, hmap branch: b + kinetic vectors n ─────────────────────
    (
        """            a = self.hmap_alpha
            qh = q.transpose(1, 2)   # (B, H, T, D)
            kh = k.transpose(1, 2)
            vh = v.transpose(1, 2)

            # Kinetic sector vectors and (optional) node potential
            m = (qh + kh) * (0.5 ** 0.5)
            g = None
""",
        """            a = self.hmap_alpha
            b = self.hmap_beta
            qh = q.transpose(1, 2)   # (B, H, T, D)
            kh = k.transpose(1, 2)
            vh = v.transpose(1, 2)

            # Kinetic sector vectors and (optional) node potential.
            # b > 0 adds the NSD kinetic part: 1/2<m,m> - b*1/2<n,n>, which at
            # b=1 is the indefinite symmetric kernel x_i^T W_S x_j. g needs no
            # change: g_i = <q_i,k_i> = 1/2||m_i||^2 - 1/2||n_i||^2 is already
            # the indefinite potential (diag sees only the symmetric sector).
            m = (qh + kh) * (0.5 ** 0.5)
            n = (qh - kh) * (0.5 ** 0.5) if b > 0.0 else None
            g = None
""",
    ),
    # ── 6. _hmap_block: signature, docstring, NSD Gram term ────────────────
    (
        """            def _hmap_block(q_blk, k_blk, m_blk, g_blk, rows):
                \"\"\"Attention output for a block of query rows against all keys.
                logits = 1/2<m,m> + (1-a)*1/2(<q,k>-<k,q>) + a*1/2(g_i - g_j).\"\"\"
                logits = 0.5 * (m_blk @ m.transpose(-2, -1))        # kinetic Gram rows
                if a < 1.0:
""",
        """            def _hmap_block(q_blk, k_blk, m_blk, n_blk, g_blk, rows):
                \"\"\"Attention output for a block of query rows against all keys.
                logits = 1/2<m,m> - b*1/2<n,n>
                       + (1-a)*1/2(<q,k>-<k,q>) + a*1/2(g_i - g_j)
                (b,a): (0,0)=AMAP, (0,1)=DMAP, (1,0)=standard, (1,1)=CMAP.\"\"\"
                logits = 0.5 * (m_blk @ m.transpose(-2, -1))        # kinetic Gram rows
                if b > 0.0:
                    logits = logits - b * 0.5 * (n_blk @ n.transpose(-2, -1))  # NSD sector
                if a < 1.0:
""",
    ),
    # ── 7. training call site ──────────────────────────────────────────────
    (
        """                y = _hmap_block(qh, kh, m, g, slice(0, T))
""",
        """                y = _hmap_block(qh, kh, m, n, g, slice(0, T))
""",
    ),
    # ── 8. evaluation chunked call site ────────────────────────────────────
    (
        """                    outs.append(_hmap_block(
                        qh[..., rows, :], kh[..., rows, :], m[..., rows, :],
                        g[..., rows] if g is not None else None, rows))
""",
        """                    outs.append(_hmap_block(
                        qh[..., rows, :], kh[..., rows, :], m[..., rows, :],
                        n[..., rows, :] if n is not None else None,
                        g[..., rows] if g is not None else None, rows))
""",
    ),
]


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("nanochat/gpt.py")
    text = path.read_text()

    if "hmap_beta" in text:
        raise SystemExit(f"{path} already contains hmap_beta — refusing to re-patch.")

    for i, (old, new) in enumerate(EDITS, 1):
        count = text.count(old)
        if count != 1:
            raise SystemExit(
                f"EDIT {i}: anchor found {count} times (need exactly 1) — "
                f"gpt.py drifted from the expected text; nothing written.\n"
                f"Anchor head: {old.splitlines()[0]!r}"
            )
        text = text.replace(old, new, 1)

    backup = path.with_suffix(path.suffix + ".pre_cmap")
    backup.write_text(path.read_text())
    path.write_text(text)
    print(f"patched {path} (backup: {backup})")
    print("config: hmap_beta in [0,1]; (beta,alpha) square: "
          "(0,0)=AMAP (0,1)=DMAP (1,0)=standard (1,1)=CMAP")
    print("sanity check to run on GPU: attn_variant='hmap', beta=1, alpha=0 "
          "must match the standard/FA3 branch logits to bf16 tolerance.")


if __name__ == "__main__":
    main()
