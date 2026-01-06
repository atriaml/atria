from __future__ import annotations

from collections.abc import Callable

from atria_logger import get_logger
from atria_ml.training.engine_steps._base import EngineStep
from atria_ml.training.engines._base import EngineBase, EngineConfig, EngineDependencies

from atria_insights.perturbation_robustness._evaluator_step import (
    PerturbationRobustnessEvaluatorStep,
)

logger = get_logger(__name__)


class PerturbationRobustnessEvaluatorEngineConfig(EngineConfig):
    pass


class PerturbationRobustnessEvaluatorEngineDependencies(EngineDependencies):
    perturbation_transform: Callable


class PerturbationRobustnessEvaluatorEngine(
    EngineBase[
        PerturbationRobustnessEvaluatorEngineConfig,
        PerturbationRobustnessEvaluatorEngineDependencies,
    ]
):
    def _build_engine_step(self) -> EngineStep:
        return PerturbationRobustnessEvaluatorStep(
            model_pipeline=self._deps.model_pipeline,
            perturbation_transform=self._deps.perturbation_transform,
            device=self._deps.device,
        )
