from __future__ import annotations

from typing import TYPE_CHECKING

from atria_logger import get_logger

from atria_ml.training.engine_steps import EngineStep, VisualizationStep
from atria_ml.training.engines._trainer_child_engine import (
    TrainerChildEngine,
    TrainerChildEngineConfig,
    TrainerChildEngineDependencies,
)

if TYPE_CHECKING:
    pass


logger = get_logger(__name__)


class VisualizationEngineDependencies(TrainerChildEngineDependencies):
    pass


class VisualizationEngineConfig(TrainerChildEngineConfig):
    pass


class VisualizationEngine(
    TrainerChildEngine[VisualizationEngineConfig, VisualizationEngineDependencies]
):
    def __init__(
        self, config: VisualizationEngineConfig, deps: VisualizationEngineDependencies
    ):
        super().__init__(config, deps)

    def _build_engine_step(self) -> EngineStep:
        return VisualizationStep(
            model_pipeline=self._deps.model_pipeline,
            device=self._deps.device,
            with_amp=self._config.with_amp,
            test_run=self._config.test_run,
            training_engine=self._deps.training_engine._engine,
        )
