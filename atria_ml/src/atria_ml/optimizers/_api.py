"""API functions for loading and preprocessing optimizers."""

from __future__ import annotations

from atria_logger import get_logger

from atria_ml.optimizers._base import OptimizerConfig
from atria_ml.registry import OPTIMIZER

logger = get_logger(__name__)


def load_optimizer_config(optimizer_name: str, **kwargs) -> OptimizerConfig:
    return OPTIMIZER.load_module_config(optimizer_name, **kwargs)  # type: ignore
