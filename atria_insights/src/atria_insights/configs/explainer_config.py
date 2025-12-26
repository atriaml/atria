from __future__ import annotations

from atria_datasets import load_dataset_config
from atria_logger import get_logger
from atria_ml.configs import (
    DataConfig,
    RuntimeEnvConfig,
    TaskConfigBase,
    TrainingTaskConfig,
)
from atria_ml.training._configs import LoggingConfig
from pydantic import ConfigDict

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
from atria_insights.model_pipelines._api import load_x_model_pipeline_config
from atria_insights.model_pipelines._common import (
    ExplainableModelPipelineConfig,
    ExplanationTargetStrategy,
)

logger = get_logger(__name__)


class ExplanationTaskConfig(TaskConfigBase):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, frozen=True, use_enum_values=True
    )
    x_model_pipeline: ExplainableModelPipelineConfig
    enable_outputs_caching: bool = False
    max_training_baseline_features: int = 100

    @classmethod
    def from_training_task_config(
        cls,
        training_task_config: TrainingTaskConfig,
        dataset_name: str | None = None,
        exp_name: str = "img_cls_00",
        output_dir: str = "./outputs",
        logging: LoggingConfig = LoggingConfig(refresh_rate=1, logging_steps=1),
        feature_segmentor: FeatureSegmentorConfigType = NoOpSegmenterConfig(),
        baseline_generator: BaselineGeneratorConfigType = SimpleBaselineGeneratorConfig(),
        explainer: ExplainerConfigType = SaliencyExplainerConfig(),
        explainability_metrics: dict[str, ExplainabilityMetricConfig] | None = None,
        explanation_target_strategy: ExplanationTargetStrategy = (
            ExplanationTargetStrategy.predicted
        ),
        iterative_computation: bool = False,
        max_training_baseline_features: int = 100,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        internal_batch_size: int = 1,
        grad_batch_size: int = 1,
    ) -> ExplanationTaskConfig:
        if dataset_name is not None:
            data = DataConfig(
                dataset_config=load_dataset_config(dataset_name),
                num_workers=num_workers,
                eval_batch_size=eval_batch_size,
            )
            if data != training_task_config.data:
                logger.warning(
                    "The provided dataset_name results in a different DataConfig than the one in the TrainingRunConfig. Using the new DataConfig."
                )
        else:
            data = training_task_config.data.model_copy(
                update={"eval_batch_size": eval_batch_size, "num_workers": num_workers}
            )
        assert training_task_config.env.model_name is not None, (
            "Model name must be specified in the TrainingRunConfig."
        )
        return ExplanationTaskConfig(
            env=RuntimeEnvConfig(
                project_name="docinsights",
                exp_name=exp_name,
                dataset_name=data.dataset_config.dataset_name.replace("/", "_"),
                model_name=training_task_config.env.model_name.replace("/", "_"),
                output_dir=output_dir,
                seed=training_task_config.env.seed,
            ),
            data=data,
            x_model_pipeline=load_x_model_pipeline_config(
                training_task_config.model_pipeline.name,
                model_pipeline=training_task_config.model_pipeline,
                feature_segmentor=feature_segmentor,
                baseline_generator=baseline_generator,
                explainer=explainer,
                explainability_metrics=explainability_metrics,
                explanation_target_strategy=explanation_target_strategy,
                iterative_computation=iterative_computation,
                internal_batch_size=internal_batch_size,
                grad_batch_size=grad_batch_size,
            ),
            max_training_baseline_features=max_training_baseline_features,
            logging=logging,
        )
