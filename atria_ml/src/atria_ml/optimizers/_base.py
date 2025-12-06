from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, TypeVar

from atria_registry import ModuleConfig

if TYPE_CHECKING:
    import torch


class OptimizerConfig(ModuleConfig):
    __builds_with_kwargs__: bool = True
    lr: float = 0.01

    def build(  # type: ignore
        self, parameters: Iterable[torch.nn.Parameter], **kwargs
    ) -> torch.optim.Optimizer:
        return super().build(params=parameters, **kwargs)


T_OptimizerConfig = TypeVar("T_OptimizerConfig", bound=OptimizerConfig)
