from __future__ import annotations

from pathlib import Path
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
        explainer: Explainer | None = None,
        device: torch.device | str = "cpu",
        persist_to_disk: bool = True,
        cache_dir: str | Path | None = None,
    ) -> Explainer:
        return super().build(
            model=model,
            explainer=explainer,
            device=device,
            persist_to_disk=persist_to_disk,
            cache_dir=cache_dir,
        )


T_ExplainabilityMetricConfig = TypeVar(
    "T_ExplainabilityMetricConfig", bound=ExplainabilityMetricConfig
)
