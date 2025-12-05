from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from atria_logger import get_logger
from ignite.engine import Engine

from atria_ml.training.configs.early_stopping_config import EarlyStoppingConfig
from atria_ml.training.configs.model_checkpoint import ModelCheckpointConfig
from atria_ml.training.engine_builders._base import EngineBase
from atria_ml.training.engine_builders._training import TrainingEngine
from atria_ml.training.engine_steps import EngineStep, TestStep, VisualizationStep
from atria_ml.training.engine_steps._evaluation import PredictStep, ValidationStep
from atria_ml.training.handlers.ema_handler import EMAHandler

if TYPE_CHECKING:
    from ignite.engine import Engine, State
    from ignite.handlers import TensorboardLogger
    from ignite.metrics import Metric

if TYPE_CHECKING:
    from ignite.engine import Engine
    from ignite.handlers import TensorboardLogger


logger = get_logger(__name__)


class TrainerEvaluationEngine(EngineBase):
    def __init__(
        self,
        *args,
        training_engine: TrainingEngine,
        run_as_child_every_n_epochs: float = 1.0,
        run_as_child_on_start: bool = True,
        use_ema_for_val: bool = False,
        with_amp: bool = False,
        ema_handler: EMAHandler | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._training_engine = training_engine
        self._run_every_n_epochs = run_as_child_every_n_epochs
        self._run_on_start = run_as_child_on_start
        self._use_ema_for_val = use_ema_for_val
        self._with_amp = with_amp
        self._ema_handler = ema_handler
        self.attach_to_training_engine()

    def attach_tb_logger(self, engine: Engine, tb_logger: TensorboardLogger):
        import ignite.distributed as idist
        from ignite.engine import Events
        from ignite.handlers import global_step_from_engine

        if idist.get_rank() == 0 and tb_logger is not None and self._logging.log_to_tb:
            tb_logger.attach_output_handler(
                engine,
                event_name=Events.EPOCH_COMPLETED,
                metric_names="all",
                tag="epoch",
                global_step_transform=global_step_from_engine(
                    self._training_engine._engine
                ),
            )

            @engine.on(Events.TERMINATE | Events.INTERRUPT | Events.EXCEPTION_RAISED)
            def on_terminate(engine: Engine) -> None:
                tb_logger.close()

    def _build_engine_step(self) -> EngineStep:
        return ValidationStep(
            model_pipeline=self._model_pipeline,
            device=self._device,
            with_amp=self._with_amp,
            test_run=self._test_run,
        )

    def attach_to_training_engine(self):
        from ignite.engine import Events

        if self._run_every_n_epochs >= 1:
            cond = Events.EPOCH_COMPLETED(every=int(self._run_every_n_epochs))
            cond = cond | Events.COMPLETED
            if self._run_on_start:
                cond = cond | Events.STARTED
            self._training_engine._engine.add_event_handler(cond, self.run)
        else:
            cond = Events.ITERATION_COMPLETED(
                every=int(
                    self._run_every_n_epochs * self._training_engine.steps_per_epoch
                )
            )
            cond = cond | Events.COMPLETED
            if self._run_on_start:
                cond = cond | Events.STARTED
            self._training_engine._engine.add_event_handler(cond, self.run)

    def run(self, checkpoint_path: str | Path | None = None) -> State:
        if self._use_ema_for_val:
            if self._ema_handler is None:
                logger.warning(
                    "EMA handler is not set. You must pass an "
                    "EMA handler to `attach_to_engine` to use ema for validation."
                )
            else:
                self._ema_handler.swap_params()
        state = super().run(checkpoint_path=checkpoint_path)
        if self._use_ema_for_val:
            if self._ema_handler is not None:
                self._ema_handler.swap_params()
        return state


class ValidationEngine(TrainerEvaluationEngine):
    def __init__(
        self,
        *args,
        early_stopping: EarlyStoppingConfig = EarlyStoppingConfig(),
        model_checkpoint_config: ModelCheckpointConfig = ModelCheckpointConfig(),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._early_stopping = early_stopping
        self._model_checkpoint_config = model_checkpoint_config

    def _build_engine_step(self) -> EngineStep:
        assert self._training_engine is not None, (
            "Parent engine must be set for ValidationEngine."
        )
        return ValidationStep(
            model_pipeline=self._model_pipeline,
            device=self._device,
            with_amp=self._with_amp,
            test_run=self._test_run,
            training_engine=self._training_engine._engine,
        )

    def attach_handlers(self, engine: Engine, stage: str):
        engine = super().attach_handlers(engine, stage)
        self.attach_early_stopping_callback(engine=engine)
        if self._model_checkpoint_config.enabled:
            self.attach_best_checkpointer()
        return engine

    def attach_early_stopping_callback(self, engine: Engine) -> None:
        if self._early_stopping.enabled:
            from ignite.engine import Events
            from ignite.handlers import Checkpoint, EarlyStopping

            es_handler = EarlyStopping(
                patience=self._early_stopping.patience,
                score_function=Checkpoint.get_default_score_fn(
                    self._early_stopping.monitored_metric,
                    -1 if self._early_stopping.mode == "min" else 1.0,
                ),
                trainer=self._training_engine._engine,
            )
            self._engine.add_event_handler(Events.COMPLETED, es_handler)

    def attach_best_checkpointer(self, engine: Engine) -> None:
        from ignite.engine import Events
        from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine

        if self._model_checkpoint_config.monitored_metric is None:
            raise ValueError(
                "Monitored metric must be specified for best model checkpointing."
            )

        checkpoint_state_dict = self._training_engine._to_load_checkpoint()

        checkpoint_dir = Path(self._output_dir) / self._model_checkpoint_config.dir
        save_handler = DiskSaver(checkpoint_dir, require_empty=False)

        logger.info(
            f"Configuring best model checkpointing with monitored metric:\n\t{self._model_checkpoint_config.monitored_metric}"
        )
        best_model_saver = Checkpoint(
            checkpoint_state_dict,
            save_handler=save_handler,
            filename_prefix="best",
            n_saved=self._model_checkpoint_config.n_best_saved,
            global_step_transform=global_step_from_engine(
                self._training_engine._engine
            ),
            score_name=self._model_checkpoint_config.monitored_metric.replace("/", "-"),
            score_function=Checkpoint.get_default_score_fn(
                self._model_checkpoint_config.monitored_metric,
                -1 if self._model_checkpoint_config.mode == "min" else 1.0,
            ),
            include_self=True,
        )
        engine.add_event_handler(Events.COMPLETED, best_model_saver)


class VisualizationEngine(TrainerEvaluationEngine):
    def _build_engine_step(self) -> EngineStep:
        return VisualizationStep(
            model_pipeline=self._model_pipeline,
            device=self._device,
            with_amp=self._with_amp,
            test_run=self._test_run,
            training_engine=self._training_engine._engine,
        )

    def attach_handlers(
        self,
        engine: Engine,
        stage: str,
        metrics: dict[str, Metric] | None = None,
        tb_logger: TensorboardLogger | None = None,
        event_handlers: list[tuple[Any, Callable]] | None = None,
    ):
        # configure engine
        self.attach_progress_bar(engine=engine, stage=stage)
        if tb_logger is not None:
            self.attach_tb_logger(engine=engine, tb_logger=tb_logger)
        if self._test_run:
            self.setup_test_run(engine=engine)
        return engine


class TestEngine(EngineBase):
    def __init__(
        self,
        *args,
        save_model_outputs_to_disk: bool = False,
        checkpoint_types: list[str] | None = None,
        with_amp: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._save_model_outputs_to_disk = save_model_outputs_to_disk
        self._checkpoint_types = checkpoint_types or ["last", "best"]
        self._with_amp = with_amp
        for key in self._checkpoint_types:
            assert key in ["last", "best"], (
                f"Checkpoint type {key} is not supported. Possible types are ['last', 'best']"
            )

    def _build_engine_step(self) -> EngineStep:
        return TestStep(
            model_pipeline=self._model_pipeline,
            device=self._device,
            with_amp=self._with_amp,
            test_run=self._test_run,
        )

    def attach_model_output_saver(self, engine: Engine, output_name: str):
        from ignite.engine import Events

        from atria_ml.training.utilities.model_output_saver import ModelOutputSaver

        if self._save_model_outputs_to_disk:
            engine.add_event_handler(
                Events.ITERATION_COMPLETED,
                ModelOutputSaver(
                    output_dir=Path(self._output_dir), output_name=output_name
                ),
            )

    def _find_checkpoint(self, checkpoint_type: str) -> str | None:
        import glob
        import os

        checkpoint_dir = Path(self._output_dir) / "checkpoints"

        if not checkpoint_dir.exists():
            return None

        available_checkpoints = glob.glob(str(checkpoint_dir) + "/*.pt")

        if checkpoint_type == "last":
            available_checkpoints = [
                c for c in available_checkpoints if "best" not in c and "last" in c
            ]
        elif checkpoint_type == "best":
            available_checkpoints = [c for c in available_checkpoints if "best" in c]
        else:
            raise ValueError(f"Unknown checkpoint type: {checkpoint_type}")

        if len(available_checkpoints) > 0:
            return max(available_checkpoints, key=os.path.getctime)
        return None

    def run_with_checkpoint_type(self, checkpoint_type: str) -> State | None:
        checkpoint_path = self._find_checkpoint(checkpoint_type)
        if checkpoint_path is None:
            raise ValueError(
                f"No {checkpoint_type} checkpoint found for testing. "
                "Please make sure that the checkpoint exists."
            )
        checkpoint_name = Path(checkpoint_path).name.replace("=", "_")
        if self._metric_logging_prefix:
            self._metric_logging_prefix += "/" + checkpoint_name
        else:
            self._metric_logging_prefix = checkpoint_name
        return super().run(checkpoint_path=checkpoint_path)


class PredictionEngine(EngineBase):
    def __init__(self, *args, with_amp: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._with_amp = with_amp

    def _build_engine_step(self) -> EngineStep:
        return PredictStep(
            model_pipeline=self._model_pipeline,
            device=self._device,
            with_amp=self._with_amp,
            test_run=self._test_run,
        )
