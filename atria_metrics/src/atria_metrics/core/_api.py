"""API functions for loading metrics ."""

from __future__ import annotations

from typing import TYPE_CHECKING

from atria_logger import get_logger
from pydantic import TypeAdapter
from atria_types import DatasetSplitType

from atria_metrics.core._base import MetricConfig
from atria_metrics.core._registry_group import METRICS

if TYPE_CHECKING:
    import torch

logger = get_logger(__name__)


def load_metric_config(
    metric_name: str, **kwargs
) -> MetricConfig:
    return  METRICS.load_module_config(metric_name, **kwargs)
