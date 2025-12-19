"""API functions for loading metrics ."""

from __future__ import annotations

from typing import TYPE_CHECKING

from atria_logger import get_logger
from atria_types import DatasetSplitType

from atria_metrics.core._base import MetricConfig
from atria_metrics.core._registry_group import METRICS

if TYPE_CHECKING:
    import torch

logger = get_logger(__name__)


def load_metric_config(
    device: torch.device | str | None = None,
    num_classes: int | None = None,
    split: DatasetSplitType | None = None,
) -> MetricConfig:
    return METRICS.load_module_config(
        device=device, num_classes=num_classes, split=split
    )  # type: ignore
