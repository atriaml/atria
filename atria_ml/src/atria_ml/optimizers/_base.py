from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, TypeVar

from atria_registry import ModuleConfig

if TYPE_CHECKING:
    import torch


class OptimizerConfig(ModuleConfig):
    __builds_with_kwargs__: bool = True
    type: str
    lr: float = 0.01

    @property
    def kwargs(self) -> dict[str, Any]:
        return self.model_dump(exclude={"module_path", "type"})

    def build(  # type: ignore
        self, parameters: Iterable[torch.nn.Parameter], **kwargs
    ) -> torch.optim.Optimizer:
        return super().build(params=parameters, **kwargs)


T_OptimizerConfig = TypeVar("T_OptimizerConfig", bound=OptimizerConfig)
