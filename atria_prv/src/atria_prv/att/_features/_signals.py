from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

SignalFn = Callable[
    [torch.Tensor, torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]
]


def _valid_mask(labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
    return labels != ignore_index


def _safe_labels(labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
    """Clamp ignore_index positions to a valid class id so gather()/index ops don't error."""
    return labels.clamp(min=0)


def token_loss(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = logits.detach().float()
    labels = labels.detach()
    B, T = labels.shape
    C = logits.shape[-1]

    # always cut down logits to labels seq length
    logits = logits[:, :T, :]
    ce = F.cross_entropy(
        logits.reshape(-1, C),
        labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).reshape(B, T)
    return ce, _valid_mask(labels, ignore_index)


def token_gold_prob(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T = labels.shape
    # always cut down logits to labels seq length
    logits = logits[:, :T, :]
    logits = logits.detach().float()
    labels = labels.detach()
    probs = F.softmax(logits, dim=-1)
    gold_prob = probs.gather(
        -1, _safe_labels(labels, ignore_index).unsqueeze(-1)
    ).squeeze(-1)
    return gold_prob, _valid_mask(labels, ignore_index)


def token_margin(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T = labels.shape
    # always cut down logits to labels seq length
    logits = logits[:, :T, :]
    logits = logits.detach().float()
    labels = labels.detach()
    probs = F.softmax(logits, dim=-1)
    k = min(2, probs.shape[-1])
    top_probs = probs.topk(k=k, dim=-1).values
    margin = top_probs[..., 0] - top_probs[..., 1] if k == 2 else top_probs[..., 0]
    return margin, _valid_mask(labels, ignore_index)


def token_entropy(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T = labels.shape
    # always cut down logits to labels seq length
    logits = logits[:, :T, :]
    logits = logits.detach().float()
    labels = labels.detach()
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy, _valid_mask(labels, ignore_index)


def token_scaled_conf_carlini(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T = labels.shape
    # always cut down logits to labels seq length
    logits = logits[:, :T, :]
    logits = logits.detach().float()
    labels = labels.detach()
    probs = F.softmax(logits, dim=-1)
    gold_prob = probs.gather(
        -1, _safe_labels(labels, ignore_index).unsqueeze(-1)
    ).squeeze(-1)
    return torch.log(gold_prob / (1 - gold_prob)), _valid_mask(labels, ignore_index)


SIGNAL_FUNCS: dict[str, SignalFn] = {
    "loss": token_loss,
    "prob": token_gold_prob,
    "scaled_conf": token_scaled_conf_carlini,
    "margin": token_margin,
    "entropy": token_entropy,
}
