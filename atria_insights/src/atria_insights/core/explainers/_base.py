from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from atria_registry import ModuleConfig

if TYPE_CHECKING:
    import torch
    from torchxai.explainers import Explainer


class ExplainerConfig(ModuleConfig):
    __builds_with_kwargs__: bool = True
    internal_batch_size: int = 64
    grad_batch_size: int = 64

    def build(  # type: ignore
        self, model: torch.nn.Module, multi_target: bool = False, **kwargs
    ) -> Explainer:
        return super().build(model=model, multi_target=multi_target, **kwargs)


T_ExplainerConfig = TypeVar("T_ExplainerConfig", bound=ExplainerConfig)
