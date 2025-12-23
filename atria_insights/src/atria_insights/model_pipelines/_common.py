from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Any, TypeVar

from atria_models import ModelPipelineConfig
from atria_registry import ModuleConfig

from atria_insights.explainability_metrics._base import ExplainabilityMetricConfig
from atria_insights.explainers._torchxai import (
    ExplainerConfigType,
    SaliencyExplainerConfig,
)
from atria_insights.model_pipelines._feature_segmentors import (
    FeatureSegmentorConfigType,
    NoOpSegmenterConfig,
)
from atria_insights.model_pipelines.baseline_generators import (
    BaselineGeneratorConfigType,
)
from atria_insights.model_pipelines.baseline_generators._simple import (
    SimpleBaselineGeneratorConfig,
)

if TYPE_CHECKING:
    from atria_insights.model_pipelines._model_pipeline import ExplainableModelPipeline


class ExplanationTargetStrategy(str, enum.Enum):
    predicted = "predicted"
    ground_truth = "ground_truth"
    all = "all"


class ExplainableModelPipelineConfig(ModuleConfig):
    __hash_exclude__: set[str] = {"explainability_metrics", "iterative_computation"}
    model_pipeline: ModelPipelineConfig
    feature_segmentor: FeatureSegmentorConfigType = NoOpSegmenterConfig()
    baseline_generator: BaselineGeneratorConfigType = SimpleBaselineGeneratorConfig()
    explainer: ExplainerConfigType = SaliencyExplainerConfig()
    explainability_metrics: dict[str, ExplainabilityMetricConfig] | None = None  #
    explanation_target_strategy: ExplanationTargetStrategy = (
        ExplanationTargetStrategy.predicted
    )
    iterative_computation: bool = False

    def build(self, **kwargs: Any) -> ExplainableModelPipeline:
        labels = kwargs.pop("labels")
        assert labels is not None, (
            "Labels must be provided to build the model pipeline."
        )
        return super().build(labels=labels, **kwargs)


T_ExplainableModelPipelineConfig = TypeVar(
    "T_ExplainableModelPipelineConfig", bound=ExplainableModelPipelineConfig
)
