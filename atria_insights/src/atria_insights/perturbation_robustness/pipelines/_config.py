from __future__ import annotations

from typing import Any, TypeVar

from atria_models import ModelPipelineConfig
from atria_transforms.core._tfs._base import DataTransform

from atria_insights.baseline_generators import BaselineGeneratorConfigType
from atria_insights.baseline_generators._simple import SimpleBaselineGeneratorConfig
from atria_insights.feature_segmentors import FeatureSegmentorConfigType
from atria_insights.feature_segmentors._base import NoOpSegmenterConfig


class FeaturePerturbationTransform(DataTransform):
    model_pipeline: ModelPipelineConfig
    feature_segmentor: FeatureSegmentorConfigType = NoOpSegmenterConfig()
    baseline_generator: BaselineGeneratorConfigType = SimpleBaselineGeneratorConfig()
    percent_features_perturbed: float = 0.5

    def __call__(self, input: Any) -> Any | list:
        return super().__call__(input)


T_PerturbationRobustnessPipelineConfig = TypeVar(
    "T_PerturbationRobustnessPipelineConfig", bound=PerturbationRobustnessPipelineConfig
)
