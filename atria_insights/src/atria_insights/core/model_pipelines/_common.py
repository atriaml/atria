from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Any, TypeVar

from atria_models import ModelPipelineConfig
from atria_registry import ModuleConfig

from atria_insights.core.explainability_metrics._base import ExplainabilityMetricConfig
from atria_insights.core.explainers._torchxai import (
    ExplainerConfigType,
    SaliencyExplainerConfig,
)
from atria_insights.core.model_pipelines._baselines_generators import (
    BaselinesGeneratorConfig,
)
from atria_insights.core.model_pipelines._feature_segmentors import (
    FeatureSegmentorConfigType,
    NoOpSegmenterConfig,
)

if TYPE_CHECKING:
    from atria_insights.core.model_pipelines._model_pipeline import (
        ExplainableModelPipeline,
    )


class ExplanationTargetStrategy(str, enum.Enum):
    predicted = "predicted"
    ground_truth = "ground_truth"
    all = "all"


class ExplainableModelPipelineConfig(ModuleConfig):
    model_pipeline_config: ModelPipelineConfig = ModelPipelineConfig()
    feature_segmentor_config: FeatureSegmentorConfigType = NoOpSegmenterConfig()
    baseline_generator_config: BaselinesGeneratorConfig = BaselinesGeneratorConfig()
    baselines_fixed_value: float = 0.5
    explainer_config: ExplainerConfigType = SaliencyExplainerConfig()
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
