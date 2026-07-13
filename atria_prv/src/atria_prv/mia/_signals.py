from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

MAX_SEQ_LEN = 512


def per_document_loss(
    logits: torch.Tensor, token_labels: torch.Tensor, *, ignore_index: int = -100
) -> np.ndarray:
    logits = logits.detach().float()
    token_labels = token_labels.detach()
    B, T, C = logits.shape

    pad_len = MAX_SEQ_LEN - token_labels.shape[1]
    if pad_len > 0:
        token_labels = F.pad(token_labels, (0, pad_len), value=-100)

    ce = F.cross_entropy(
        logits.reshape(-1, C),
        token_labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).reshape(B, T)

    mask = token_labels != ignore_index
    valid = mask.sum(dim=1).clamp(min=1)
    loss = (ce * mask).sum(dim=1) / valid
    return loss.cpu().numpy().astype("float32")
