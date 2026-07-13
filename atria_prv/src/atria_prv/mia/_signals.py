from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def per_document_signals(
    logits: torch.Tensor,
    token_labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce a batch of token-classification outputs to per-document MIA signals.

    The token-classification head only returns a batch-mean scalar loss, so for a
    membership inference attack we recompute per-document signals directly from the
    logits and labels, masking the ``-100`` ignore index used by the head
    (``atria_models/.../_heads/_token_classification.py``).

    Args:
        logits: ``[B, T, C]`` token logits.
        token_labels: ``[B, T]`` token labels (``-100`` marks ignored positions).
        ignore_index: label value marking positions to ignore.
        reduction: ``"mean"`` averages the per-token loss over valid tokens;
            ``"sum"`` returns the summed loss.

    Returns:
        Tuple of numpy arrays:
        - ``p_doc`` ``[B, C]``: mean softmax distribution over valid tokens
          (the ART "prediction" feature per document).
        - ``y_doc`` ``[B]`` (int64): per-document majority true label (ART ``y``).
        - ``loss_doc`` ``[B]``: per-document cross-entropy (kept for logging).
    """
    logits = logits.detach().float()
    token_labels = token_labels.detach()
    B, T, C = logits.shape

    mask = token_labels != ignore_index
    valid = mask.sum(dim=1).clamp(min=1)

    # per-document mean softmax distribution over valid tokens -> ART features
    probs = logits.softmax(dim=-1)
    p_doc = (probs * mask.unsqueeze(-1)).sum(dim=1) / valid.unsqueeze(-1)

    # per-document majority-ish true label -> ART y (used for one-hot / rule-based)
    y_doc = (token_labels.clamp(min=0) * mask).sum(dim=1) // valid

    # per-document cross-entropy loss (masked) -> kept for logging / future loss-based attacks
    ce = F.cross_entropy(
        logits.reshape(-1, C),
        token_labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).reshape(B, T)
    denom = valid if reduction == "mean" else torch.ones_like(valid)
    loss_doc = (ce * mask).sum(dim=1) / denom

    return (
        p_doc.cpu().numpy().astype("float32"),
        y_doc.cpu().numpy().astype("int64"),
        loss_doc.cpu().numpy().astype("float32"),
    )
