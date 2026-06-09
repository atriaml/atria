from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from atria_registry import ModuleConfig

from atria_insights.storage.sample_cache_managers._metric_data_cacher import (
    MetricDataCacher,
)

if TYPE_CHECKING:
    import torch
    from torchxai.explainers import Explainer


class ExplainabilityMetricConfig(ModuleConfig):
    type: str
    enabled: bool = False

    def build(  # type: ignore
        self,
        model: torch.nn.Module,
        explainer: Explainer | None = None,
        device: torch.device | str = "cpu",
        cacher: MetricDataCacher | None = None,
    ) -> Explainer:
        name = self.type.split("/")[-1]
        return super().build(
            model=model,
            explainer=explainer,
            device=device,
            cacher=cacher,
            metric_key="-".join([name, self.hash]),
        )


T_ExplainabilityMetricConfig = TypeVar(
    "T_ExplainabilityMetricConfig", bound=ExplainabilityMetricConfig
)
