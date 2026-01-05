from __future__ import annotations

from typing import TypeVar

from atria_models import ModelPipelineConfig
from atria_registry import ModuleConfig

from atria_insights.baseline_generators import BaselineGeneratorConfigType
from atria_insights.baseline_generators._simple import SimpleBaselineGeneratorConfig
from atria_insights.feature_segmentors import FeatureSegmentorConfigType
from atria_insights.feature_segmentors._base import NoOpSegmenterConfig


class FeaturePerturbationPipelineConfig(ModuleConfig):
    model_pipeline: ModelPipelineConfig
    feature_segmentor: FeatureSegmentorConfigType = NoOpSegmenterConfig()
    baseline_generator: BaselineGeneratorConfigType = SimpleBaselineGeneratorConfig()
    percent_features_perturbed: float = 0.5


T_FeaturePerturbationPipelineConfig = TypeVar(
    "T_FeaturePerturbationPipelineConfig", bound=FeaturePerturbationPipelineConfig
)
