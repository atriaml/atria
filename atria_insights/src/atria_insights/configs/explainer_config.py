from __future__ import annotations

from atria_datasets import load_dataset_config
from atria_logger import get_logger
from atria_ml.configs import (
    DataConfig,
    RuntimeEnvConfig,
    TaskConfigBase,
    TrainingTaskConfig,
)
from pydantic import ConfigDict

from atria_insights.explainability_metrics._base import ExplainabilityMetricConfig
from atria_insights.explainers._torchxai import (
    ExplainerConfigType,
    SaliencyExplainerConfig,
)
from atria_insights.model_pipelines._api import load_x_model_pipeline_config
from atria_insights.model_pipelines._common import (
    ExplainableModelPipelineConfig,
    ExplanationTargetStrategy,
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

logger = get_logger(__name__)


class ExplanationTaskConfig(TaskConfigBase):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, frozen=True, use_enum_values=True
    )
    x_model_pipeline: ExplainableModelPipelineConfig
    enable_outputs_caching: bool = False

    @classmethod
    def from_training_task_config(
        cls,
        config: TrainingTaskConfig,
        dataset_name: str | None = None,
        exp_name: str = "img_cls_00",
        output_dir: str = "./outputs",
        feature_segmentor: FeatureSegmentorConfigType = NoOpSegmenterConfig(),
        baseline_generator: BaselineGeneratorConfigType = SimpleBaselineGeneratorConfig(),
        explainer: ExplainerConfigType = SaliencyExplainerConfig(),
        explainability_metrics: dict[str, ExplainabilityMetricConfig] | None = None,
        explanation_target_strategy: ExplanationTargetStrategy = (
            ExplanationTargetStrategy.predicted
        ),
        iterative_computation: bool = False,
        enable_outputs_caching: bool = False,
    ) -> ExplanationTaskConfig:
        if dataset_name is not None:
            data = DataConfig(
                dataset_config=load_dataset_config(dataset_name), num_workers=8
            )
            if data != config.data:
                logger.warning(
                    "The provided dataset_name results in a different DataConfig than the one in the TrainingRunConfig. Using the new DataConfig."
                )
        else:
            data = config.data
        assert config.env.model_name is not None, (
            "Model name must be specified in the TrainingRunConfig."
        )
        return ExplanationTaskConfig(
            env=RuntimeEnvConfig(
                project_name="docinsights",
                exp_name=exp_name,
                dataset_name=data.dataset_config.dataset_name.replace("/", "_"),
                model_name=config.env.model_name.replace("/", "_"),
                output_dir=output_dir,
                seed=config.env.seed,
            ),
            data=data,
            x_model_pipeline=load_x_model_pipeline_config(
                config.model_pipeline.name,
                model_pipeline=config.model_pipeline,
                feature_segmentor=feature_segmentor,
                baseline_generator=baseline_generator,
                explainer=explainer,
                explainability_metrics=explainability_metrics,
                explanation_target_strategy=explanation_target_strategy,
                iterative_computation=iterative_computation,
            ),
            enable_outputs_caching=enable_outputs_caching,
        )
