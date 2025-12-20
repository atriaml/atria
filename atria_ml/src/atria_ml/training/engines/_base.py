from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import field
from functools import partial
from pathlib import Path
from typing import Any, Generic, TypeVar

import torch
from atria_logger import get_logger
from atria_models.core.model_pipelines._common import ModelPipelineConfig
from atria_models.core.model_pipelines._model_pipeline import ModelPipeline
from ignite.engine import Engine, State
from ignite.handlers import TensorboardLogger
from ignite.metrics import Metric
from pydantic import BaseModel, ConfigDict

from atria_ml.training._configs import LoggingConfig
from atria_ml.training.engine_steps import EngineStep
from atria_ml.training.engines.utilities import (
    _extract_output,
    _format_metrics_for_logging,
)

logger = get_logger(__name__)


class EngineDependencies(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)
    model_pipeline: ModelPipeline[ModelPipelineConfig]
    dataloader: torch.utils.data.DataLoader
    device: str | torch.device
    output_dir: str | Path
    tb_logger: TensorboardLogger | None = None
    event_handlers: list[tuple[Any, Callable]] | None = None


class EngineConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_epochs: int = 1
    epoch_length: int | None = None
    outputs_to_running_avg: list[str] = field(default_factory=lambda: ["loss"])
    logging: LoggingConfig = LoggingConfig()
    metric_logging_prefix: str | None = None
    test_run: bool = False
    use_fixed_batch_iterator: bool = False
    with_amp: bool = False


T_EngineConfig = TypeVar("T_EngineConfig", bound=EngineConfig)
T_EngineDependencies = TypeVar("T_EngineDependencies", bound=EngineDependencies)


class EngineBase(Generic[T_EngineConfig, T_EngineDependencies]):
    def __init__(self, config: T_EngineConfig, deps: T_EngineDependencies):
        self._config = config
        self._deps = deps
        self._metrics: dict[str, Metric] | None = None
        self._engine_step, self._engine = self._build_engine()
        self._metrics = self._deps.model_pipeline.build_metrics(
            stage=self._engine_step.name, device=self._deps.device
        )
        self._attach_handlers()

    @property
    def batches_per_epoch(self) -> int:
        return len(self._deps.dataloader)

    @property
    def steps_per_epoch(self) -> int:
        return self.batches_per_epoch

    @property
    def total_update_steps(self) -> int:
        return self.steps_per_epoch * self._config.max_epochs

    def _build_engine_step(self) -> EngineStep:
        """
        Abstract method for engine step. Must be implemented by subclasses.
        """
        raise NotImplementedError

    def _initialize_ignite_engine(self, engine_step: EngineStep) -> Engine:
        # initialize the Ignite engine
        engine = Engine(engine_step)
        engine.logger.propagate = False
        return engine

    def _build_engine(self) -> tuple[EngineStep, Engine]:
        # initialize the engine step
        engine_step = self._build_engine_step()

        # initialize the Ignite engine
        engine = self._initialize_ignite_engine(engine_step=engine_step)

        return engine_step, engine

    def _attach_handlers(self) -> None:
        from ignite.engine import Events

        # configure engine
        self._setup_test_run()
        self._attach_profilers()
        self._attach_event_handlers()
        self._attach_metrics()

        # loggers must come last
        self._attach_progress_bar()
        self._attach_tb_logger()

        @self._engine.on(Events.TERMINATE | Events.INTERRUPT)
        def on_terminate(engine: Engine) -> None:
            logger.info(
                f"Engine [{self.__class__.__name__}] terminated after {engine.state.epoch} epochs."
            )

        # add handler for exception
        @self._engine.on(Events.EXCEPTION_RAISED)
        def on_exception(exception: Exception) -> None:
            raise exception

    def _attach_progress_bar(self) -> None:
        import ignite.distributed as idist
        from ignite.engine import Events
        from ignite.handlers import ProgressBar

        # initialize the progress bar
        progress_bar = ProgressBar(
            desc=f"Stage [{self._engine_step.name}]", persist=True
        )

        if idist.get_rank() == 0:
            progress_bar.attach(
                self._engine,
                event_name=Events.ITERATION_COMPLETED(
                    every=self._config.logging.refresh_rate
                ),
                metric_names="all",
            )

            def _log_eval_metrics(logger, epoch, elapsed, tag, metrics):
                logger.info(
                    "Epoch %d - Evaluation time: %.2fs - %s metrics: EpochResult:",
                    epoch,
                    elapsed,
                    tag,
                )
                formatted_metrics = _format_metrics_for_logging(metrics)
                logger.info(f"{self._engine_step.name} metrics:")
                logger.info(json.dumps(formatted_metrics, indent=4))

            @self._engine.on(Events.EPOCH_COMPLETED)
            def progress_on_epoch_completed(engine: Engine) -> None:
                _log_eval_metrics(
                    logger=logger,
                    epoch=engine.state.epoch,
                    elapsed=engine.state.times["EPOCH_COMPLETED"],
                    tag=self._engine_step.name,
                    metrics=engine.state.metrics,
                )

            @self._engine.on(Events.TERMINATE | Events.INTERRUPT)
            def on_terminate(engine: Engine) -> None:
                progress_bar.close()

    def _attach_tb_logger(self):
        import ignite.distributed as idist
        from ignite.engine import Events

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
            )

            @self._engine.on(
                Events.TERMINATE | Events.INTERRUPT | Events.EXCEPTION_RAISED
            )
            def on_terminate(engine: Engine) -> None:
                if self._deps.tb_logger is not None:
                    self._deps.tb_logger.close()

    def _attach_profilers(self):
        if self._config.logging.profile_time:
            from ignite.handlers import BasicTimeProfiler, HandlersTimeProfiler

            HandlersTimeProfiler().attach(self._engine)
            BasicTimeProfiler().attach(self._engine)

    def _setup_test_run(self):
        from ignite.engine import Events

        if not self._config.test_run:
            return

        logger.warning(
            f"This is a test run of engine [{self.__class__.__name__}]. "
            "Only a single engine step will be executed."
        )

        def terminate_on_iteration_complete(
            engine,
        ):  # this is necessary for fldp to work with correct privacy accounting
            logger.info("Terminating engine as test_run=True")
            engine.terminate()

        def print_iteration_started_info(engine):
            logger.debug(
                f"Batch input received for engine [{self.__class__.__name__}]:"
            )
            logger.debug(engine.state.batch)

        def print_iteration_completed_info(engine):
            logger.debug(f"Output received for engine [{self.__class__.__name__}]:")
            logger.debug(engine.state.output)

        self._engine.add_event_handler(
            Events.ITERATION_COMPLETED, terminate_on_iteration_complete
        )
        self._engine.add_event_handler(
            Events.ITERATION_STARTED, print_iteration_started_info
        )
        self._engine.add_event_handler(
            Events.ITERATION_COMPLETED, print_iteration_completed_info
        )

    def _attach_metrics(self) -> None:
        import ignite.distributed as idist
        import torch
        from ignite.metrics import GpuInfo, RunningAverage
        from ignite.metrics.metric import EpochWise

        for index, key in enumerate(self._config.outputs_to_running_avg):
            RunningAverage(
                alpha=0.98,
                output_transform=partial(_extract_output, index=index, key=key),
                epoch_bound=True,
            ).attach(self._engine, f"{self._engine_step.name}/running_avg_{key}")

        if self._config.logging.log_gpu_stats:
            if idist.device() != torch.device("cpu"):
                GpuInfo().attach(self._engine, name="gpu")

        if self._metrics is not None and len(self._metrics) > 0:
            for metric_name, metric in self._metrics.items():
                logger.info(
                    f"Attaching metric '{metric_name}' to engine '{self.__class__.__name__}'"
                )
                metric.attach(
                    self._engine,
                    (
                        f"{self._engine_step.name}/{metric_name}"
                        if self._config.metric_logging_prefix is None
                        else f"{self._engine_step.name}/{self._config.metric_logging_prefix}/{metric_name}"
                    ),
                    usage=EpochWise(),
                )

    def _attach_event_handlers(self):
        if self._deps.event_handlers is None:
            return
        for event, handler in self._deps.event_handlers:
            self._engine.add_event_handler(event, handler)

    def _load_state_dict(
        self, engine: Engine, save_weights_only: bool = False
    ) -> dict[str, Any]:
        from atria_ml.training.engines.utilities import MODEL_PIPELINE_CHECKPOINT_KEY

        checkpoint_state_dict = {
            MODEL_PIPELINE_CHECKPOINT_KEY: self._deps.model_pipeline
        }

        return checkpoint_state_dict

    def _to_load_state_dict(self) -> dict[str, Any]:
        from atria_ml.training.engines.utilities import MODEL_PIPELINE_CHECKPOINT_KEY

        checkpoint_state_dict = {
            MODEL_PIPELINE_CHECKPOINT_KEY: self._deps.model_pipeline
        }

        return checkpoint_state_dict

    def _load_checkpoint(self, checkpoint_path: str | Path) -> None:
        import torch
        from ignite.handlers.checkpoint import Checkpoint

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        Checkpoint.load_objects(
            to_load=self._to_load_state_dict(), checkpoint=checkpoint, strict=True
        )

    def run(self, checkpoint_path: str | Path | None = None) -> State | None:
        from atria_ml.training.engines.utilities import FixedBatchIterator

        # run engine
        if self._deps.output_dir is not None:
            logger.info(
                f"Running {self.__class__.__name__} engine with batch size [{self._deps.dataloader.batch_size}] and output_dir: {self._deps.output_dir}"
            )
        else:
            logger.info(f"Running engine {self.__class__.__name__}.")

        # move model pipeline to device
        self._deps.model_pipeline.ops.to_device(self._deps.device)

        # load checkpoint if provided
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path=checkpoint_path)

        return self._engine.run(
            (
                FixedBatchIterator(
                    self._deps.dataloader, self._deps.dataloader.batch_size
                )
                if self._config.use_fixed_batch_iterator
                else self._deps.dataloader
            ),
            max_epochs=self._config.max_epochs,
            epoch_length=self._config.epoch_length,
        )
