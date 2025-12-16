from __future__ import annotations

import enum
from typing import Any, TypeVar

from atria_models import ModelPipelineConfig
from atria_registry import ModuleConfig

from atria_insights.core.explainability_metrics._base import ExplainabilityMetricConfig
from atria_insights.core.explainers._torchxai import ExplainerConfigType
from atria_insights.core.model_pipelines._baselines_generators import (
    BaselinesGeneratorConfig,
)
from atria_insights.core.model_pipelines._feature_segmentors import (
    FeatureSegmentorConfigType,
)
from atria_insights.core.model_pipelines._model_pipeline import ExplainableModelPipeline


class ExplanationTargetStrategy(str, enum.Enum):
    predicted = "predicted"
    ground_truth = "ground_truth"
    all = "all"


class ExplainableModelPipelineConfig(ModuleConfig):
    model_pipeline_config: ModelPipelineConfig
    feature_segmentor_config: FeatureSegmentorConfigType
    baseline_generator_config: BaselinesGeneratorConfig = BaselinesGeneratorConfig()
    baselines_fixed_value: float | None = None
    explainer_config: ExplainerConfigType
    explainability_metrics: dict[str, ExplainabilityMetricConfig] | None = None  #
    explanation_target_strategy: ExplanationTargetStrategy = (
        ExplanationTargetStrategy.predicted
    )
    iterative_computation: bool = False

    def build(self, **kwargs: Any) -> ExplainableModelPipeline:
        model_pipeline = self.model_pipeline_config.build(**kwargs)
        return super().build(model_pipeline=model_pipeline, **kwargs)


T_ExplainableModelPipelineConfig = TypeVar(
    "T_ExplainableModelPipelineConfig", bound=ExplainableModelPipelineConfig
)
