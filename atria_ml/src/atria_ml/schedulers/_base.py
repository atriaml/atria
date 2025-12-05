from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from atria_registry import ModuleConfig

if TYPE_CHECKING:
    import torch


class LRSchedulerConfig(ModuleConfig):
    __builds_with_kwargs__: bool = True

    def build(  # type: ignore
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler._LRScheduler:
        return super().build(optimizer=optimizer)


T_LRSchedulerConfig = TypeVar("T_LRSchedulerConfig", bound=LRSchedulerConfig)
