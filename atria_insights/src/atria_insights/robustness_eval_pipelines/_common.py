from __future__ import annotations

from typing import TypeVar

from atria_models import ModelPipelineConfig
from atria_registry import ModuleConfig

from atria_insights.baseline_generators import BaselineGeneratorConfigType
from atria_insights.baseline_generators._simple import SimpleBaselineGeneratorConfig


class RobustnessEvalModelPipelineConfig(ModuleConfig):
    model_pipeline: ModelPipelineConfig
    baseline_generator: BaselineGeneratorConfigType = SimpleBaselineGeneratorConfig()


T_RobustnessEvalModelPipelineConfig = TypeVar(
    "T_RobustnessEvalModelPipelineConfig", bound=RobustnessEvalModelPipelineConfig
)
