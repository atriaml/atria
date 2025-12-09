from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from atria_registry import ModuleConfig

if TYPE_CHECKING:
    import torch
    from ignite.metrics import Metric


class MetricConfig(ModuleConfig):
    __builds_with_kwargs__: bool = True

    def build(  # type: ignore
        self, device: torch.device | str | None = None, num_classes: int | None = None
    ) -> Metric:
        return super().build(device=device, num_classes=num_classes)


T_MetricConfig = TypeVar("T_MetricConfig", bound=MetricConfig)
