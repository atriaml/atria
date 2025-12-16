from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from atria_logger import get_logger

from atria_ml.task_pipelines._utilities import _find_checkpoint
from atria_ml.training.engine_steps import EngineStep, TestStep
from atria_ml.training.engines._base import EngineBase, EngineConfig, EngineDependencies
from atria_ml.training.engines._exceptions import NoCheckpointFoundError

if TYPE_CHECKING:
    from ignite.engine import State


logger = get_logger(__name__)


class TestEngineConfig(EngineConfig):
    save_model_outputs_to_disk: bool = False


class TestEngineDependencies(EngineDependencies):
    pass


class TestEngine(EngineBase[TestEngineConfig, TestEngineDependencies]):
    def __init__(self, config: TestEngineConfig, deps: TestEngineDependencies) -> None:
        super().__init__(config=config, deps=deps)

    def _build_engine_step(self) -> EngineStep:
        return TestStep(
            model_pipeline=self._deps.model_pipeline,
            device=self._deps.device,
            with_amp=self._config.with_amp,
            test_run=self._config.test_run,
        )

    def attach_model_output_saver(self):
        from ignite.engine import Events

        from atria_ml.training.utilities.model_output_saver import ModelOutputSaver

        if self._config.save_model_outputs_to_disk:
            self._engine.add_event_handler(
                Events.ITERATION_COMPLETED,
                ModelOutputSaver(output_dir=Path(self._deps.output_dir)),
            )

    def run_with_checkpoint_type(self, checkpoint_type: str) -> State | None:
        checkpoint_path = _find_checkpoint(
            output_dir=self._deps.output_dir, checkpoint_type=checkpoint_type
        )
        if checkpoint_path is None:
            if checkpoint_type == "last":
                logger.warning(
                    "No last checkpoint found for testing. "
                    "Running test engine without loading a checkpoint."
                )
            else:
                raise NoCheckpointFoundError(
                    f"No {checkpoint_type} checkpoint found for testing. "
                    "Please make sure that the checkpoint exists."
                )
        else:
            checkpoint_name = Path(checkpoint_path).name.replace("=", "_")
            self._config = self._config.model_copy(
                update={
                    "metric_logging_prefix": self._config.metric_logging_prefix
                    + "/"
                    + checkpoint_name
                    if self._config.metric_logging_prefix
                    else checkpoint_name
                }
            )
        return super().run(checkpoint_path=checkpoint_path)
