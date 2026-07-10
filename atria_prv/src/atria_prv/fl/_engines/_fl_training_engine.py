from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from atria_logger import get_logger
from atria_ml.configs._task import TrainingTaskConfig
from atria_ml.training._configs import ModelCheckpointConfig
from atria_ml.training.engine_steps import EngineStep
from atria_ml.training.engines._base import EngineBase, EngineConfig, EngineDependencies

from atria_prv.fl._engines._fl_training_step import FLTrainingStep
from atria_prv.fl.configs import FLClientTrainingTaskConfig

if TYPE_CHECKING:
    from ignite.engine import State

logger = get_logger(__name__)


class FLTrainingEngineConfig(EngineConfig):
    total_num_clients: int = 2
    client_fraction: float = 1.0
    seed: int = 42
    model_checkpoint: ModelCheckpointConfig = ModelCheckpointConfig()


class FLTrainingEngineDependencies(EngineDependencies):
    run_config: TrainingTaskConfig
    # the client trainers whose local updates this engine aggregates each round
    client_training_task_configs: list[FLClientTrainingTaskConfig]


class FLTrainingEngine(
    EngineBase[FLTrainingEngineConfig, FLTrainingEngineDependencies]
):
    """Federated-learning training engine.

    Unlike ``TrainerEngine`` there are no optimizers, LR schedulers, warmup, EMA or
    gradient accumulation -- in FL the FedAvg *aggregation is the update*. So this
    inherits the plain ``EngineBase`` (which already gives us the metrics/progress-bar/
    TB-logger plumbing over epochs == rounds) and adds back only the two things an FL
    round needs: model checkpointing and the validation-engine hook.
    """

    _config: FLTrainingEngineConfig
    _deps: FLTrainingEngineDependencies

    def __init__(
        self, config: FLTrainingEngineConfig, deps: FLTrainingEngineDependencies
    ) -> None:
        self._config = config
        self._deps = deps
        self._validation_engine = None
        self._metrics = None
        self._engine_step, self._engine = self._build_engine()
        self._attach_handlers()

    def _build_engine_step(self) -> EngineStep:
        return FLTrainingStep(
            model_pipeline=self._deps.model_pipeline,
            device=self._deps.device,
            client_training_task_configs=self._deps.client_training_task_configs,
            total_num_clients=self._config.total_num_clients,
            client_fraction=self._config.client_fraction,
            seed=self._config.seed,
            with_amp=self._config.with_amp,
            test_run=self._config.test_run,
        )

    def attach_validation_engine(self, validation_engine) -> None:
        self._validation_engine = validation_engine

    def _attach_handlers(self) -> None:
        super()._attach_handlers()
        self.attach_model_checkpointer()

    def _to_save_state_dict(self) -> dict[str, Any]:
        from atria_ml.training.engines.utilities import (
            CONFIG_KEY,
            MODEL_PIPELINE_CHECKPOINT_KEY,
            TRAINING_ENGINE_KEY,
        )

        return {
            CONFIG_KEY: self._deps.run_config,
            MODEL_PIPELINE_CHECKPOINT_KEY: self._deps.model_pipeline,
            TRAINING_ENGINE_KEY: self._engine,
        }

    def _to_load_state_dict(self) -> dict[str, Any]:
        from atria_ml.training.engines.utilities import (
            MODEL_PIPELINE_CHECKPOINT_KEY,
            TRAINING_ENGINE_KEY,
        )

        return {
            MODEL_PIPELINE_CHECKPOINT_KEY: self._deps.model_pipeline,
            TRAINING_ENGINE_KEY: self._engine,
        }

    def attach_model_checkpointer(self) -> None:
        from ignite.engine import Events
        from ignite.handlers import DiskSaver
        from ignite.handlers.checkpoint import BaseSaveHandler, Checkpoint

        if not self._config.model_checkpoint.enabled:
            return

        logger.info("Configuring model checkpointing with the following config:")
        logger.info(f"{self._config.model_checkpoint}")
        checkpoint_state_dict = self._to_save_state_dict()

        checkpoint_dir = Path(self._deps.output_dir) / self._config.model_checkpoint.dir
        save_handler = DiskSaver(checkpoint_dir, require_empty=False)
        if self._config.model_checkpoint.save_per_epoch:
            checkpoint_handler = Checkpoint(
                checkpoint_state_dict,
                cast(Callable | BaseSaveHandler, save_handler),
                filename_prefix=self._config.model_checkpoint.name_prefix,
                global_step_transform=lambda *_: self._engine.state.epoch,
                n_saved=self._config.model_checkpoint.n_saved,
            )
            self._engine.add_event_handler(
                Events.EPOCH_COMPLETED(
                    every=self._config.model_checkpoint.save_every_iters
                ),
                checkpoint_handler,
            )
        else:
            checkpoint_handler = Checkpoint(
                checkpoint_state_dict,
                cast(Callable | BaseSaveHandler, save_handler),
                filename_prefix=self._config.model_checkpoint.name_prefix,
                n_saved=self._config.model_checkpoint.n_saved,
            )
            self._engine.add_event_handler(
                Events.ITERATION_COMPLETED(
                    every=self._config.model_checkpoint.save_every_iters
                )
                | Events.COMPLETED,
                checkpoint_handler,
            )

    def run(self, checkpoint_path: str | Path | None = None) -> State:
        from atria_ml.task_pipelines._utilities import _find_checkpoint

        if (
            checkpoint_path is None
            and self._config.model_checkpoint.resume_from_checkpoint
        ):
            checkpoint_path = _find_checkpoint(
                output_dir=self._deps.output_dir, checkpoint_type="last"
            )
        return super().run(checkpoint_path=checkpoint_path)
