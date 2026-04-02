from __future__ import annotations

import torch
from torchxai.data_types import ExplanationTarget


class AttentionTokenTarget(ExplanationTarget):
    """
    Target a specific output token position — identical semantics to
    post-hoc targets, just indexing into the sequence dimension.

    indices: one token position per batch item. Use 0 for [CLS], or any position for token-level saliency.
    """

    indices: list[list[int]]

    def select(self, attn: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attn: (B, L, L) — head-reduced
        Returns:
            (B, T, L_k)
        """
        B = attn.shape[0]
        assert len(self.indices) == B, (
            f"Target batch size {len(self.indices)} != input batch size {B}"
        )
        return torch.stack([attn[b, self.indices[b], :] for b in range(B)])

    @property
    def value(self) -> list[int]:
        return self.indices
