from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from atria_logger import get_logger

from atria_ml.training.engine_steps import EngineStep
from atria_ml.training.engine_steps._evaluation import ValidationStep
from atria_ml.training.engines._base import EngineBase, EngineConfig, EngineDependencies
from atria_ml.training.engines._trainer import TrainerEngine

if TYPE_CHECKING:
    from ignite.engine import State


logger = get_logger(__name__)


class TrainerChildEngineDependencies(EngineDependencies):
    training_engine: TrainerEngine


class TrainerChildEngineConfig(EngineConfig):
    run_every_n_epochs: float = 1.0
    run_on_start: bool = True
    use_ema: bool = False


T_TrainerChildEngineConfig = TypeVar(
    "T_TrainerChildEngineConfig", bound=TrainerChildEngineConfig
)
T_TrainerChildEngineDependencies = TypeVar(
    "T_TrainerChildEngineDependencies", bound=TrainerChildEngineDependencies
)


class TrainerChildEngine(
    EngineBase[T_TrainerChildEngineConfig, T_TrainerChildEngineDependencies]
):
    def __init__(
        self, config: T_TrainerChildEngineConfig, deps: T_TrainerChildEngineDependencies
    ):
        super().__init__(config=config, deps=deps)
        self.attach_to_training_engine()

    def _attach_tb_logger(self):
        import ignite.distributed as idist
        from ignite.engine import Events
        from ignite.handlers import global_step_from_engine

        if (
            idist.get_rank() == 0
            and self._deps.tb_logger is not None
            and self._config.logging.log_to_tb
        ):
            self._deps.tb_logger.attach_output_handler(
                self._engine,
                event_name=Events.EPOCH_COMPLETED,
                metric_names="all",
                tag="epoch",
                global_step_transform=global_step_from_engine(
                    self._deps.training_engine._engine
                ),
            )

    def _build_engine_step(self) -> EngineStep:
        return ValidationStep(
            model_pipeline=self._deps.model_pipeline,
            device=self._deps.device,
            with_amp=self._config.with_amp,
            test_run=self._config.test_run,
        )

    def attach_to_training_engine(self):
        from ignite.engine import Events

        if self._config.run_every_n_epochs >= 1:
            cond = Events.EPOCH_COMPLETED(every=int(self._config.run_every_n_epochs))
            cond = cond | Events.COMPLETED
            if self._config.run_on_start:
                cond = cond | Events.STARTED
            self._deps.training_engine._engine.add_event_handler(cond, self.run)
        else:
            cond = Events.ITERATION_COMPLETED(
                every=int(
                    self._config.run_every_n_epochs
                    * self._deps.training_engine.steps_per_epoch
                )
            )
            cond = cond | Events.COMPLETED
            if self._config.run_on_start:
                cond = cond | Events.STARTED
            self._deps.training_engine._engine.add_event_handler(cond, self.run)

    def run(self) -> State | None:
        if self._config.use_ema:
            if self._deps.training_engine.ema_handler is None:
                logger.warning(
                    "EMA handler is not set. You must pass an "
                    "EMA handler to `attach_to_engine` to use ema for validation."
                )
            else:
                self._deps.training_engine.ema_handler.swap_params()
        state = super().run(checkpoint_path=None)
        if self._config.use_ema and self._deps.training_engine.ema_handler is not None:
            self._deps.training_engine.ema_handler.swap_params()
        return state
