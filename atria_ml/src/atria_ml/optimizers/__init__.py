# ruff: noqa

from typing import TYPE_CHECKING

import lazy_loader as lazy

if TYPE_CHECKING:
    from atria_ml.optimizers._api import load_optimizer_config
    from atria_ml.optimizers._base import OptimizerConfig
    from atria_ml.optimizers._registry_group import OPTIMIZERS
    from atria_ml.optimizers._configs import (
        AdadeltaOptimizerConfig,
        AdagradOptimizerConfig,
        AdamOptimizerConfig,
        AdamWOptimizerConfig,
        LARSOptimizerConfig,
        RMSpropOptimizerConfig,
        SGDOptimizerConfig,
    )

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_api": ["load_optimizer_config"],
        "_base": ["OptimizerConfig"],
        "_registry_group": ["OPTIMIZERS"],
        "_torch": [
            "AdadeltaOptimizerConfig",
            "AdagradOptimizerConfig",
            "AdamOptimizerConfig",
            "AdamWOptimizerConfig",
            "LARSOptimizerConfig",
            "RMSpropOptimizerConfig",
            "SGDOptimizerConfig",
        ],
    },
)
