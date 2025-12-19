"""API functions for loading and preprocessing optimizers."""

from __future__ import annotations

from atria_logger import get_logger

from atria_ml.schedulers._base import LRSchedulerConfig

logger = get_logger(__name__)


def load_lr_scheduler_config(sch_name: str, **kwargs) -> LRSchedulerConfig:
    from atria_ml.schedulers._registry_group import LR_SCHEDULERS

    return LR_SCHEDULERS.load_module_config(sch_name, **kwargs)  # type: ignore
