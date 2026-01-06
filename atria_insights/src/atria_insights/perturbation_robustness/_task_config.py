from __future__ import annotations

from atria_logger import get_logger
from atria_ml.configs import TaskConfigBase
from atria_models.core.model_pipelines._common import ModelPipelineConfig
from pydantic import ConfigDict

logger = get_logger(__name__)


class PerturbationRobustnessEvaluatorTaskConfig(TaskConfigBase):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, frozen=True, use_enum_values=True
    )
    model_pipeline: ModelPipelineConfig
    n_runs_per_perturbation: int = 5
