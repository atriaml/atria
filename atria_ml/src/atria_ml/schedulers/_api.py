"""API functions for loading and preprocessing optimizers."""

from __future__ import annotations

from atria_logger import get_logger

from atria_ml.registry import LR_SCHEDULER
from atria_ml.schedulers._base import LRSchedulerConfig

logger = get_logger(__name__)


def load_lr_scheduler_config(sch_name: str, **kwargs) -> LRSchedulerConfig:
    return LR_SCHEDULER.load_module_config(sch_name, **kwargs)  # type: ignore
