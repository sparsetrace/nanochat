"""
In-context learning (ICL) and induction-head evaluations.

Both metrics follow Olsson et al., "In-context Learning and Induction Heads"
(2022), which studied models at exactly this scale.

1) icl_score: loss as a function of position in context on ordinary validation
   data. Models that learn in context predict later tokens better than early
   ones; the score is  loss(late window) - loss(early window)  — more negative
   means more in-context learning. Watch for the phase change early in training
   when induction circuitry forms.

2) induction_score: sequences of uniformly random tokens are repeated
   ([s ; s]). The first half is unpredictable by construction (loss ~ log V);
   the second half is perfectly predictable iff the model can perform the
   induction operation: "find the previous occurrence of this token, copy what
   followed." This isolates the mechanism from all linguistic confounds.
   Reported: mean loss on each half, and argmax accuracy on the second half.

Both are pure forward passes (no kv-cache), so they work for all attention
variants including HMAP; the eager HMAP path evaluates through its chunked
no-grad branch. AR-objective metrics — automatically skipped in MLM mode
(they run under the eval_every gate, which MLM disables).
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def icl_score(model, val_loader, num_batches=8, early=(40, 60), late=(490, 510)):
    """Per-position loss on validation batches; ICL score = late - early.

    val_loader must yield (x, y) token batches (as nanochat's val loader does),
    with y == -1 at positions excluded from the loss.
    """
    pos_sums = None
    pos_counts = None
    for _ in range(num_batches):
        x, y = next(val_loader)
        B, T = x.shape
        loss = model(x, y, loss_reduction='none').view(B, T)  # fp32
        valid = (y.view(B, T) != -1).to(loss.dtype)
        if pos_sums is None:
            pos_sums = torch.zeros(T, device=loss.device)
            pos_counts = torch.zeros(T, device=loss.device)
        pos_sums += (loss * valid).sum(dim=0)
        pos_counts += valid.sum(dim=0)
    per_pos = pos_sums / pos_counts.clamp(min=1.0)
    early_loss = per_pos[early[0]:early[1]].mean().item()
    late_loss = per_pos[late[0]:late[1]].mean().item()
    return {"early": early_loss, "late": late_loss, "score": late_loss - early_loss}


@torch.no_grad()
def induction_score(model, vocab_size, device, seq_len=256, batch_size=16,
                    num_batches=4, seed=1234):
    """Repeated-random-sequence induction eval.

    Builds [s ; s] with s ~ Uniform(vocab)^seq_len, runs next-token prediction,
    and scores the two halves separately. Fixed seed => identical eval set
    across steps, runs, and attention variants (comparable numbers).
    """
    g = torch.Generator().manual_seed(seed)
    agg = {"random_half_loss": 0.0, "induction_loss": 0.0, "induction_acc": 0.0}
    for _ in range(num_batches):
        s = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g).to(device)
        seq = torch.cat([s, s], dim=1)          # (B, 2L)
        x, y = seq[:, :-1], seq[:, 1:]          # next-token setup
        logits = model(x)                       # (B, 2L-1, vocab), fp32 softcapped
        losses = F.cross_entropy(
            logits.transpose(1, 2).float(), y, reduction='none')  # (B, 2L-1)
        preds = logits.argmax(dim=-1)
        L = seq_len
        # positions 0..L-2 predict within the first (random) copy;
        # positions L..2L-2 predict within the second (predictable) copy.
        agg["random_half_loss"] += losses[:, :L - 1].mean().item()
        agg["induction_loss"] += losses[:, L:].mean().item()
        agg["induction_acc"] += (preds[:, L:] == y[:, L:]).float().mean().item()
    return {k: v / num_batches for k, v in agg.items()}
