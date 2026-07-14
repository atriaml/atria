from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from atria_prv.att._features._bio_scheme import BioScheme
from atria_prv.att._features._signals import SignalFn


@dataclass
class AggConfig:
    percentiles: tuple[float, ...] = (10, 25, 50, 75, 90)
    n_bins: int = 10
    top_k_frac: float = 0.1
    hist_ranges: dict[str, tuple[float, float]] | None = None
    empty_split_fill: float = 0.0
    include_count_features: bool = True


class TokenSignalExtractor:
    """Reduces per-token signals to a fixed-size, named feature vector per sample.

    For every ``signal x BIO-split`` pair, computes mean/std/min/max/bottom-k-mean/
    top-k-mean/percentiles/normalized-histogram over the tokens in that split, plus a
    ``count`` of contributing tokens. Splits with zero valid tokens (e.g. a document
    with no entities) get ``empty_split_fill`` for every stat and ``count=0``, so the
    attack model can learn to discount them instead of confusing "no entities" with
    "entities scoring exactly 0".
    """

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
        self._hist_ranges = self._resolve_hist_ranges()

    @property
    def config(self) -> AggConfig:
        return self._config

    @property
    def signal_names(self) -> list[str]:
        return sorted(self._signals.keys())

    def _resolve_hist_ranges(self) -> dict[str, tuple[float, float]]:
        max_entropy = math.log(self._num_labels) if self._num_labels else 1.0
        max_rank = float(max(self._num_labels - 1, 1)) if self._num_labels else 1.0
        defaults = {
            "loss": (0.0, 10.0),
            "prob": (0.0, 1.0),
            "margin": (0.0, 1.0),
            "entropy": (0.0, max_entropy),
            "rank": (0.0, max_rank),
        }
        if self._config.hist_ranges:
            defaults.update(self._config.hist_ranges)
        return defaults

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
        hist_range = self._hist_ranges.get(signal_name, (0.0, 1.0))

        stat_names = ["mean", "std", "min", "max", "bottom_k_mean", "top_k_mean"] + [
            f"p{int(p)}" for p in cfg.percentiles
        ]
        hist_names = [f"hist_{i}" for i in range(cfg.n_bins)]
        out: dict[str, np.ndarray] = {
            f"{prefix}__{name}": np.full(B, cfg.empty_split_fill, dtype=np.float32)
            for name in stat_names + hist_names
        }
        if cfg.include_count_features:
            out[f"{prefix}__count"] = np.zeros(B, dtype=np.float32)

        for i in range(B):
            row_values = values[i][mask[i]]
            n = row_values.shape[0]
            if cfg.include_count_features:
                out[f"{prefix}__count"][i] = n
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

            for p in cfg.percentiles:
                out[f"{prefix}__p{int(p)}"][i] = np.percentile(row_values, p)

            hist, _ = np.histogram(row_values, bins=cfg.n_bins, range=hist_range)
            hist_normalized = hist.astype(np.float32) / n
            for b in range(cfg.n_bins):
                out[f"{prefix}__hist_{b}"][i] = hist_normalized[b]

        return out
