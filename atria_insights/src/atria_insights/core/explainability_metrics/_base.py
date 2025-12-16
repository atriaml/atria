from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from atria_registry import ModuleConfig

if TYPE_CHECKING:
    import torch
    from torchxai.explainers import Explainer


class ExplainabilityMetricConfig(ModuleConfig):
    __builds_with_kwargs__: bool = True

    def build(  # type: ignore
        self,
        model: torch.nn.Module,
        with_amp: bool = False,
        device: torch.device | str = "cpu",
        explainer: Explainer | None = None,
    ) -> Explainer:
        return super().build(
            model=model, explainer=explainer, with_amp=with_amp, device=device
        )


T_ExplainabilityMetricConfig = TypeVar(
    "T_ExplainabilityMetricConfig", bound=ExplainabilityMetricConfig
)
