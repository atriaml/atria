from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from atria_registry import ModuleConfig

from atria_insights.baseline_generators import BaselineGeneratorConfigType
from atria_insights.baseline_generators._simple import SimpleBaselineGeneratorConfig
from atria_insights.feature_segmentors import FeatureSegmentorConfigType
from atria_insights.feature_segmentors._image import GridSegmenterConfig

if TYPE_CHECKING:
    import torch
    from torchxai.explainers import Explainer


class ExplainabilityMetricConfig(ModuleConfig):
    type: str
    baselines_generator: BaselineGeneratorConfigType = SimpleBaselineGeneratorConfig()
    feature_segmentor: FeatureSegmentorConfigType = GridSegmenterConfig()

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
