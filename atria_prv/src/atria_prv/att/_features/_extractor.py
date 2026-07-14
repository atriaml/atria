from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from atria_prv.att._features._bio_scheme import BioScheme
from atria_prv.att._features._signals import SignalFn


@dataclass
class AggConfig:
    top_k_frac: float = 0.1


class TokenSignalExtractor:
    def __init__(
        self,
        signals: dict[str, SignalFn],
        bio_scheme: BioScheme,
        config: AggConfig,
        ignore_index: int = -100,
        num_labels: int | None = None,
    ) -> None:
        self._signals = signals
        self._bio_scheme = bio_scheme
        self._config = config
        self._ignore_index = ignore_index
        self._num_labels = num_labels

    @property
    def config(self) -> AggConfig:
        return self._config

    @property
    def signal_names(self) -> list[str]:
        return sorted(self._signals.keys())

    def raw_signals(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        return {
            name: fn(logits, labels, self._ignore_index)
            for name, fn in self._signals.items()
        }

    def to_features(self, logits: torch.Tensor, labels: torch.Tensor) -> pd.DataFrame:
        """[B, F] one row per sample, columns named '{signal}__{split}__{agg}'."""
        raw = self.raw_signals(logits, labels)
        splits = self._bio_scheme.splits(labels, self._ignore_index)

        columns: dict[str, np.ndarray] = {}
        for signal_name, (values, valid_mask) in raw.items():
            values_np = values.cpu().numpy()
            for split_name, split_mask in splits.items():
                mask_np = (valid_mask & split_mask).cpu().numpy()
                prefix = f"{signal_name}__{split_name}"
                columns.update(self._aggregate(values_np, mask_np, prefix, signal_name))
        return pd.DataFrame(columns)

    def _aggregate(
        self, values: np.ndarray, mask: np.ndarray, prefix: str, signal_name: str
    ) -> dict[str, np.ndarray]:
        B = values.shape[0]
        cfg = self._config
        stat_names = ["mean", "std", "min", "max", "bottom_k_mean", "top_k_mean"]
        out: dict[str, np.ndarray] = {
            f"{prefix}__{name}": np.full(B, 0.0, dtype=np.float32)
            for name in stat_names
        }

        for i in range(B):
            row_values = values[i][mask[i]]
            n = row_values.shape[0]
            if n == 0:
                continue

            out[f"{prefix}__mean"][i] = row_values.mean()
            out[f"{prefix}__std"][i] = row_values.std()
            out[f"{prefix}__min"][i] = row_values.min()
            out[f"{prefix}__max"][i] = row_values.max()

            sorted_values = np.sort(row_values)
            k = max(1, int(round(n * cfg.top_k_frac)))
            out[f"{prefix}__bottom_k_mean"][i] = sorted_values[:k].mean()
            out[f"{prefix}__top_k_mean"][i] = sorted_values[-k:].mean()

        return out
