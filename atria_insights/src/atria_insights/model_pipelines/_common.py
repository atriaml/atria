from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Any, ClassVar, Self, TypeVar

from atria_models import ModelPipelineConfig
from atria_registry import ModuleConfig
from pydantic import model_validator

from atria_insights.baseline_generators import BaselineGeneratorConfigType
from atria_insights.baseline_generators._simple import SimpleBaselineGeneratorConfig
from atria_insights.explainability_metrics._base import ExplainabilityMetricConfig
from atria_insights.explainers._torchxai import (
    ExplainerConfigType,
    SaliencyExplainerConfig,
)
from atria_insights.feature_segmentors import (
    FeatureSegmentorConfigType,
    NoOpSegmenterConfig,
)

if TYPE_CHECKING:
    from atria_insights.model_pipelines._model_pipeline import ExplainableModelPipeline


class ExplanationTargetStrategy(str, enum.Enum):
    predicted = "predicted"
    ground_truth = "ground_truth"
    all = "all"


class ExplainableModelPipelineConfig(ModuleConfig):
    __hash_exclude__: ClassVar[set[str]] = {
        "explainability_metrics",
        "iterative_computation",
        "internal_batch_size",
        "grad_batch_size",
    }
    model_pipeline: ModelPipelineConfig
    feature_segmentor: FeatureSegmentorConfigType = NoOpSegmenterConfig()
    baseline_generator: BaselineGeneratorConfigType = SimpleBaselineGeneratorConfig()
    explainer: ExplainerConfigType = SaliencyExplainerConfig()
    explainability_metrics: dict[str, ExplainabilityMetricConfig] | None = None  #
    explanation_target_strategy: ExplanationTargetStrategy = (
        ExplanationTargetStrategy.predicted
    )
    iterative_computation: bool = False
    internal_batch_size: int = 1
    grad_batch_size: int = 1
    persist_to_disk: bool = False
    cache_dir: str | None = None

    @model_validator(mode="after")
    def validate_explainer(self) -> Self:
        explainer_type = self.explainer.type
        if explainer_type == "grad/deeplift_shap":
            if not isinstance(
                self.baseline_generator, FeatureBasedBaselineGeneratorConfig
            ):
                raise ValueError(
                    "DeepLIFT/DeepSHAP explainer requires a FeatureBasedBaselineGeneratorConfig."
                )

        if self.persist_to_disk and not self.cache_dir:
            raise ValueError("cache_dir must be specified if persist_to_disk is True.")

        return self

    def build(self, **kwargs: Any) -> ExplainableModelPipeline:
        labels = kwargs.pop("labels")
        assert labels is not None, (
            "Labels must be provided to build the model pipeline."
        )
        return super().build(labels=labels, **kwargs)


T_ExplainableModelPipelineConfig = TypeVar(
    "T_ExplainableModelPipelineConfig", bound=ExplainableModelPipelineConfig
)
