from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Any, ClassVar, Self, TypeVar

from atria_models import ModelPipelineConfig
from atria_registry import ModuleConfig
from pydantic import BaseModel, Field, model_validator

from atria_insights.baseline_generators import BaselineGeneratorConfigType
from atria_insights.baseline_generators._feature_based import (
    FeatureBasedBaselineGeneratorConfig,
)
from atria_insights.baseline_generators._simple import SimpleBaselineGeneratorConfig
from atria_insights.explainability_metrics import ExplainabilityMetricConfigType
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


class SlidingWindowShape(BaseModel):
    key: str = "image"
    shape: list[int] = [3, 16, 16]


class ExplainableModelPipelineConfig(ModuleConfig):
    __hash_exclude__: ClassVar[set[str]] = {
        "explainability_metrics",
        "iterative_computation",
        "internal_batch_size",
        "grad_batch_size",
        "throw_on_load_mismatch",
        "profile_time",
    }
    __schema_exclude__: ClassVar[set[str]] = {
        "model_pipeline",
        "throw_on_load_mismatch",
        "profile_time",
        "metric_baseline_generator",
    }
    model_pipeline: ModelPipelineConfig
    feature_segmentor: FeatureSegmentorConfigType = NoOpSegmenterConfig()
    baseline_generator: BaselineGeneratorConfigType = SimpleBaselineGeneratorConfig()
    metric_baseline_generator: BaselineGeneratorConfigType = (
        SimpleBaselineGeneratorConfig()
    )
    # only for occlusion explainer
    sliding_window_shapes_map: list[SlidingWindowShape] = Field(
        default_factory=lambda: [SlidingWindowShape(key="image", shape=[3, 16, 16])],
        json_schema_extra={"default": [{"key": "image", "shape": [3, 16, 16]}]},
    )
    strides_map: list[SlidingWindowShape] = Field(
        default_factory=lambda: [SlidingWindowShape(key="image", shape=[3, 8, 8])],
        json_schema_extra={"default": [{"key": "image", "shape": [3, 8, 8]}]},
    )
    explainer: ExplainerConfigType = SaliencyExplainerConfig()
    explainability_metrics: dict[str, ExplainabilityMetricConfigType] | None = None  #
    explanation_target_strategy: ExplanationTargetStrategy = (
        ExplanationTargetStrategy.predicted
    )
    iterative_computation: bool = False
    internal_batch_size: int = 1
    grad_batch_size: int = 1
    throw_on_load_mismatch: bool = False
    profile_time: bool = False

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
