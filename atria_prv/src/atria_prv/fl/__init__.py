# ruff: noqa

from typing import TYPE_CHECKING

import lazy_loader as lazy

if TYPE_CHECKING:
    from ._trainers._fl_trainer import FLTrainer, FLTrainerState
    from .configs import FLConfig, FLTrainingTaskConfig, FLClientDataConfig


__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_trainers._fl_trainer": ["FLTrainer", "FLTrainerState"],
        "configs": ["FLConfig", "FLTrainingTaskConfig", "FLClientDataConfig"],
    },
)
