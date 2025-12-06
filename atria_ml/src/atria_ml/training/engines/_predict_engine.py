from __future__ import annotations

from atria_logger import get_logger

from atria_ml.training.engine_steps import EngineStep
from atria_ml.training.engine_steps._evaluation import PredictStep
from atria_ml.training.engines._base import EngineBase, EngineConfig, EngineDependencies

logger = get_logger(__name__)


class PredictionEngineConfig(EngineConfig):
    pass


class PredictionEngineDependencies(EngineDependencies):
    pass


class PredictionEngine(
    EngineBase[PredictionEngineConfig, PredictionEngineDependencies]
):
    def _build_engine_step(self) -> EngineStep:
        return PredictStep(
            model_pipeline=self._deps.model_pipeline,
            device=self._deps.device,
            with_amp=self._config.with_amp,
            test_run=self._config.test_run,
        )
