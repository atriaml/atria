from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from atria_registry import ModuleConfig
from traitlets import Any

if TYPE_CHECKING:
    import torch
    from torchxai.explainers import Explainer


class ExplainerConfig(ModuleConfig):
    __builds_with_kwargs__: bool = True

    @property
    def kwargs(self) -> dict[str, Any]:
        return self.model_dump(exclude={"module_path", "type"})

    def build(  # type: ignore
        self,
        model: torch.nn.Module,
        internal_batch_size: int = 1,
        multi_target: bool = False,
        **kwargs,
    ) -> Explainer:
        kwargs.pop("grad_batch_size")
        return super().build(
            model=model,
            internal_batch_size=internal_batch_size,
            multi_target=multi_target,
            **kwargs,
        )


T_ExplainerConfig = TypeVar("T_ExplainerConfig", bound=ExplainerConfig)
