# ruff: noqa

from typing import TYPE_CHECKING

import lazy_loader as lazy

if TYPE_CHECKING:
    from atria_ml.schedulers._api import load_lr_scheduler_config
    from atria_ml.schedulers._base import LRSchedulerConfig
    from atria_ml.schedulers._registry_group import LR_SCHEDULER
    from atria_ml.schedulers._configs import (
        CosineAnnealingLRSchedulerConfig,
        CyclicLRSchedulerConfig,
        ExponentialLRSchedulerConfig,
        MultiStepLRSchedulerConfig,
        ReduceLROnPlateauSchedulerConfig,
        StepLRSchedulerConfig,
    )


__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_api": ["load_lr_scheduler_config"],
        "_base": ["LRSchedulerConfig"],
        "_registry_group": ["LR_SCHEDULER"],
        "_torch": [
            "StepLRSchedulerConfig",
            "MultiStepLRSchedulerConfig",
            "ExponentialLRSchedulerConfig",
            "CyclicLRSchedulerConfig",
            "ReduceLROnPlateauSchedulerConfig",
            "CosineAnnealingLRSchedulerConfig",
        ],
    },
)
