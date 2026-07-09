from __future__ import annotations

from typing import TYPE_CHECKING, Any

from atria_logger import get_logger
from atria_ml.training.engine_steps import EngineStep
from atria_ml.training.engines._base import EngineBase
from atria_ml.training.engines._trainer import (
    TrainerEngine,
    TrainerEngineConfig,
    TrainerEngineDependencies,
)

from atria_prv.engines._fl_training_step import FLTrainingStep

if TYPE_CHECKING:
    from ignite.engine import Engine

logger = get_logger(__name__)


class FLEngineConfig(TrainerEngineConfig):
    total_num_clients: int = 2
    client_fraction: float = 1.0
    seed: int = 42


class FLEngineDependencies(TrainerEngineDependencies):
    # the client trainers whose local updates this engine aggregates each round
    client_trainers: list[Any]


class FLEngine(TrainerEngine):
    _config: FLEngineConfig
    _deps: FLEngineDependencies

    def _build_engine(self) -> tuple[EngineStep, Engine]:
        # FL has no optimizers/schedulers -- aggregation *is* the update -- so skip
        # TrainerEngine._build_engine (which builds + asserts optimizers) and build the
        # step + ignite engine directly. Empty dicts also make _to_save/_to_load emit no
        # opt_*/lr_sch_* keys, leaving {config, model_pipeline, training_engine}.
        self._optimizers = {}
        self._lr_schedulers = {}
        return EngineBase._build_engine(self)

    def _build_engine_step(self) -> EngineStep:
        return FLTrainingStep(
            model_pipeline=self._deps.model_pipeline,
            device=self._deps.device,
            client_trainers=self._deps.client_trainers,
            total_num_clients=self._config.total_num_clients,
            client_fraction=self._config.client_fraction,
            seed=self._config.seed,
            with_amp=self._config.with_amp,
            test_run=self._config.test_run,
        )

    def _attach_progress_bar(self) -> None:
        # TrainerEngine's progress bar references optimizer_step / ema_momentum / lr
        # schedulers, none of which exist for FL; the plain EngineBase bar (over
        # epochs == rounds) is exactly what we want.
        EngineBase._attach_progress_bar(self)
