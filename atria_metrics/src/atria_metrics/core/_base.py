from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, TypeVar

from atria_registry import ModuleConfig
from atria_registry._utilities import _resolve_module_from_path

if TYPE_CHECKING:
    import torch
    from ignite.metrics import Metric


class MetricConfig(ModuleConfig):
    @property
    def kwargs(self) -> dict[str, object]:
        return self.model_dump(exclude={"module_path", "name"})

    def build(  # type: ignore[return]
        self,
        device: torch.device | str,
        stage: Literal["train", "test", "validation"],
        num_classes: int | None = None,
    ) -> Metric:
        assert self.module_path is not None, (
            "module_path must be set to build the module."
        )
        module = _resolve_module_from_path(self.module_path)
        if isinstance(module, type | Callable):
            possible_args = inspect.signature(module.__init__).parameters
            kwargs = {
                arg: value
                for arg, value in {
                    "device": device,
                    "num_classes": num_classes,
                    "stage": stage,
                }.items()
                if arg in possible_args and value is not None
            }

            current_kwargs = self.kwargs
            current_kwargs.update(kwargs)
            return module(**current_kwargs)
        else:
            raise TypeError(
                f"Module at path {self.module_path} is neither a class nor a callable."
            )


T_MetricConfig = TypeVar("T_MetricConfig", bound=MetricConfig)
