from __future__ import annotations

from atria_logger import get_logger
from atria_ml.configs import TaskConfigBase
from pydantic import ConfigDict

from atria_insights.feature_perturbation.pipelines._config import (
    FeaturePerturbationPipelineConfig,
)

logger = get_logger(__name__)


class FeaturePerturbationConfig(TaskConfigBase):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, frozen=True, use_enum_values=True
    )
    fp_pipeline: FeaturePerturbationPipelineConfig
