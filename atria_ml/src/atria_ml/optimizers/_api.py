"""API functions for loading and preprocessing optimizers."""

from __future__ import annotations

from atria_logger import get_logger

from atria_ml.optimizers._base import OptimizerConfig

logger = get_logger(__name__)


def load_optimizer_config(optimizer_name: str, **kwargs) -> OptimizerConfig:
    from atria_ml.optimizers._registry_group import OPTIMIZER

    return OPTIMIZER.load_module_config(optimizer_name, **kwargs)  # type: ignore
