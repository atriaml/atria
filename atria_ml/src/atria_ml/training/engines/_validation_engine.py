from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from atria_logger import get_logger

from atria_ml.training._configs import EarlyStoppingConfig, ModelCheckpointConfig
from atria_ml.training.engine_steps import EngineStep
from atria_ml.training.engine_steps._evaluation import ValidationStep
from atria_ml.training.engines._trainer_child_engine import (
    TrainerChildEngine,
    TrainerChildEngineConfig,
    TrainerChildEngineDependencies,
)
from atria_ml.training.engines.utilities import _format_metrics_for_logging

if TYPE_CHECKING:
    pass


logger = get_logger(__name__)


class ValidationEngineDependencies(TrainerChildEngineDependencies):
    pass


class ValidationEngineConfig(TrainerChildEngineConfig):
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()
    model_checkpoint: ModelCheckpointConfig = ModelCheckpointConfig()


class ValidationEngine(
    TrainerChildEngine[ValidationEngineConfig, ValidationEngineDependencies]
):
    def __init__(
        self, config: ValidationEngineConfig, deps: ValidationEngineDependencies
    ):
        super().__init__(config=config, deps=deps)
        self._deps.training_engine.attach_validation_engine(self._engine)  # type: ignore

    def _build_engine_step(self) -> EngineStep:
        assert self._deps.training_engine is not None, (
            "Parent engine must be set for ValidationEngine."
        )
        return ValidationStep(
            model_pipeline=self._deps.model_pipeline,
            device=self._deps.device,
            with_amp=self._config.with_amp,
            test_run=self._config.test_run,
            training_engine=self._deps.training_engine._engine,
        )

    def _attach_handlers(self):
        engine = super()._attach_handlers()
        self.attach_early_stopping_callback()
        if self._config.model_checkpoint.enabled:
            self.attach_best_checkpointer()
        self.attach_metrics_output_file()
        return engine

    def attach_early_stopping_callback(self) -> None:
        if self._config.early_stopping.enabled:
            from ignite.engine import Events
            from ignite.handlers import Checkpoint, EarlyStopping

            es_handler = EarlyStopping(
                patience=self._config.early_stopping.patience,
                score_function=Checkpoint.get_default_score_fn(
                    self._config.early_stopping.monitored_metric,
                    -1 if self._config.early_stopping.mode == "min" else 1.0,
                ),
                trainer=self._deps.training_engine._engine,
            )
            self._engine.add_event_handler(Events.COMPLETED, es_handler)

    def attach_best_checkpointer(self) -> None:
        from ignite.engine import Events
        from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine

        if self._config.model_checkpoint.monitored_metric is None:
            raise ValueError(
                "Monitored metric must be specified for best model checkpointing."
            )

        checkpoint_state_dict = self._deps.training_engine._to_save_state_dict()
        checkpoint_dir = Path(self._deps.output_dir) / self._config.model_checkpoint.dir
        save_handler = DiskSaver(checkpoint_dir, require_empty=False)

        logger.info(
            f"Configuring best model checkpointing with monitored metric:\n\t{self._config.model_checkpoint.monitored_metric}"
        )
        best_model_saver = Checkpoint(
            checkpoint_state_dict,
            save_handler=save_handler,
            filename_prefix="best",
            n_saved=self._config.model_checkpoint.n_best_saved,
            global_step_transform=global_step_from_engine(
                self._deps.training_engine._engine
            ),
            score_name=self._config.model_checkpoint.monitored_metric.replace("/", "-"),
            score_function=Checkpoint.get_default_score_fn(
                self._config.model_checkpoint.monitored_metric,
                -1 if self._config.model_checkpoint.mode == "min" else 1.0,
            ),
            include_self=True,
        )
        self._engine.add_event_handler(Events.COMPLETED, best_model_saver)

    def attach_metrics_output_file(self) -> None:
        from ignite.engine import Engine, Events

        def log_metrics_to_file(engine: Engine) -> None:
            parent_epoch = self._deps.training_engine._engine.state.epoch
            output_file_path = (
                Path(self._deps.output_dir) / "validation" / "metrics.json"
            )

            if not output_file_path.parent.exists():
                output_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Load existing metrics if file exists
            if output_file_path.exists():
                with open(output_file_path) as f:
                    all_metrics = json.load(f)
            else:
                all_metrics = {}

            # Add current epoch metrics
            metrics = _format_metrics_for_logging(engine.state.metrics)
            all_metrics[f"epoch_{parent_epoch}"] = metrics

            logger.debug(
                f"Dumping validation metrics for epoch {parent_epoch} to {output_file_path}:\n{json.dumps(metrics, indent=4)}"
            )
            with open(output_file_path, "w") as f:
                json.dump(all_metrics, f, indent=4)
            logger.info(f"Metrics dumped to {output_file_path}")

        self._engine.add_event_handler(Events.COMPLETED, log_metrics_to_file)
