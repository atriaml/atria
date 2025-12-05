from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from atria_logger import get_logger
from atria_models.core.model_pipelines._model_pipeline import ModelPipeline

from atria_ml.training.configs.logging_config import LoggingConfig
from atria_ml.training.engine_steps import EngineStep
from atria_ml.training.engines.utilities import _extract_output

if TYPE_CHECKING:
    import torch
    from ignite.engine import Engine, State
    from ignite.handlers import TensorboardLogger
    from ignite.metrics import Metric

logger = get_logger(__name__)


class EngineBase:
    def __init__(
        self,
        model_pipeline: ModelPipeline,
        dataloader: torch.utils.data.DataLoader,
        device: str | torch.device,
        output_dir: str | Path,
        metrics: dict[str, Metric] | None = None,
        tb_logger: TensorboardLogger | None = None,
        event_handlers: list[tuple[Any, Callable]] | None = None,
        max_epochs: int = 1,
        epoch_length: int | None = None,
        outputs_to_running_avg: list[str] | None = None,
        logging: LoggingConfig = LoggingConfig(),
        metric_logging_prefix: str | None = None,
        sync_batchnorm: bool = False,
        test_run: bool = False,
        use_fixed_batch_iterator: bool = False,
    ):
        self._model_pipeline = model_pipeline
        self._dataloader = dataloader
        self._output_dir = output_dir
        self._device = device
        self._metrics = metrics
        self._tb_logger = tb_logger
        self._event_handlers = event_handlers
        self._max_epochs = max_epochs
        self._epoch_length = epoch_length
        self._outputs_to_running_avg = (
            outputs_to_running_avg if outputs_to_running_avg is not None else ["loss"]
        )
        self._logging = logging if logging is not None else LoggingConfig()
        self._metric_logging_prefix = metric_logging_prefix
        self._sync_batchnorm = sync_batchnorm
        self._test_run = test_run
        self._use_fixed_batch_iterator = use_fixed_batch_iterator
        self._engine = self.build()

    @property
    def batches_per_epoch(self) -> int:
        return len(self._dataloader)

    @property
    def steps_per_epoch(self) -> int:
        return self.batches_per_epoch

    @property
    def total_update_steps(self) -> int:
        return self.steps_per_epoch * self._max_epochs

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

    def build(self) -> Engine:
        # initialize the engine step
        engine_step = self._build_engine_step()

        # initialize the Ignite engine
        engine = self._initialize_ignite_engine(engine_step=engine_step)

        self.attach_handlers(engine=engine, stage=engine_step.stage)
        return engine

    def attach_handlers(self, engine: Engine, stage: str):
        from ignite.engine import Events

        @engine.on(Events.TERMINATE | Events.INTERRUPT)
        def on_terminate(engine: Engine) -> None:
            logger.info(
                f"Engine [{self.__class__.__name__}] terminated after {engine.state.epoch} epochs."
            )

        # configure engine
        self.attach_progress_bar(engine=engine, stage=stage)
        if self._tb_logger is not None:
            self.attach_tb_logger(engine=engine, tb_logger=self._tb_logger)
        if self._test_run:
            self.setup_test_run(engine=engine)
        if self._logging.profile_time:
            self.attach_profilers(engine=engine)
        if self._event_handlers is not None:
            self.attach_event_handlers(
                engine=engine, event_handlers=self._event_handlers
            )
        if self._metrics is not None:
            self.attach_metrics(engine=engine, stage=stage, metrics=self._metrics)
        return engine

    def attach_progress_bar(self, engine: Engine, stage: str) -> None:
        import ignite.distributed as idist
        from ignite.engine import Events
        from ignite.handlers import ProgressBar

        from atria_ml.training.engines.utilities import _log_eval_metrics

        # initialize the progress bar
        progress_bar = ProgressBar(desc=f"Stage [{stage}]", persist=True)

        if idist.get_rank() == 0:
            progress_bar.attach(
                engine,
                event_name=Events.ITERATION_COMPLETED(every=self._logging.refresh_rate),
                metric_names="all",
            )

            @engine.on(Events.EPOCH_COMPLETED)
            def progress_on_epoch_completed(engine: Engine) -> None:
                _log_eval_metrics(
                    logger=logger,
                    epoch=engine.state.epoch,
                    elapsed=engine.state.times["EPOCH_COMPLETED"],
                    tag=stage,
                    metrics=engine.state.metrics,
                )

            @engine.on(Events.TERMINATE | Events.INTERRUPT)
            def on_terminate(engine: Engine) -> None:
                progress_bar.close()

    def attach_tb_logger(self, engine: Engine, tb_logger: TensorboardLogger):
        import ignite.distributed as idist
        from ignite.engine import Events

        if idist.get_rank() == 0 and tb_logger is not None and self._logging.log_to_tb:
            tb_logger.attach_output_handler(
                engine,
                event_name=Events.EPOCH_COMPLETED,
                metric_names="all",
                tag="epoch",
            )

            @engine.on(Events.TERMINATE | Events.INTERRUPT | Events.EXCEPTION_RAISED)
            def on_terminate(engine: Engine) -> None:
                tb_logger.close()

    def attach_profilers(self, engine: Engine):
        if self._logging.profile_time:
            from ignite.handlers import BasicTimeProfiler, HandlersTimeProfiler

            HandlersTimeProfiler().attach(engine)
            BasicTimeProfiler().attach(engine)

    def setup_test_run(self, engine: Engine):
        from ignite.engine import Events

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

        engine.add_event_handler(
            Events.ITERATION_COMPLETED, terminate_on_iteration_complete
        )
        engine.add_event_handler(Events.ITERATION_STARTED, print_iteration_started_info)
        engine.add_event_handler(
            Events.ITERATION_COMPLETED, print_iteration_completed_info
        )

    def attach_metrics(
        self,
        engine: Engine,
        stage: str | None = None,
        metrics: dict[str, Metric] | None = None,
    ) -> None:
        import ignite.distributed as idist
        import torch
        from ignite.metrics import GpuInfo, RunningAverage
        from ignite.metrics.metric import EpochWise

        for index, key in enumerate(self._outputs_to_running_avg):
            RunningAverage(
                alpha=0.98,
                output_transform=partial(_extract_output, index=index, key=key),
                epoch_bound=True,
            ).attach(engine, f"{stage}/running_avg_{key}")

        if self._logging.log_gpu_stats:
            if idist.device() != torch.device("cpu"):
                GpuInfo().attach(engine, name="gpu")

        if metrics is not None and len(metrics) > 0:
            for metric_name, metric in metrics.items():
                metric.attach(
                    engine,
                    (
                        f"{stage}/{metric_name}"
                        if self._metric_logging_prefix is None
                        else f"{stage}/{self._metric_logging_prefix}/{metric_name}"
                    ),
                    usage=EpochWise(),
                )

    def attach_event_handlers(
        self, engine: Engine, event_handlers: list[tuple[Any, Callable]]
    ):
        for event, handler in event_handlers:
            engine.add_event_handler(event, handler)

    def _load_state_dict(
        self, engine: Engine, save_weights_only: bool = False
    ) -> dict[str, Any]:
        from atria_ml.training.engines.utilities import MODEL_PIPELINE_CHECKPOINT_KEY

        checkpoint_state_dict = {MODEL_PIPELINE_CHECKPOINT_KEY: self._model_pipeline}

        return checkpoint_state_dict

    def _to_load_checkpoint(self) -> dict[str, Any]:
        from atria_ml.training.engines.utilities import MODEL_PIPELINE_CHECKPOINT_KEY

        checkpoint_state_dict = {MODEL_PIPELINE_CHECKPOINT_KEY: self._model_pipeline}

        return checkpoint_state_dict

    def _load_checkpoint(self, checkpoint_path: str | Path) -> None:
        import torch
        from ignite.handlers.checkpoint import Checkpoint

        logger.info(
            f"Running engine with weights loaded from checkpoint: {checkpoint_path}"
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        to_load = self._to_load_checkpoint()
        Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint, strict=False)

    def run(self, checkpoint_path: str | Path | None = None) -> State:
        from atria_ml.training.engines.utilities import FixedBatchIterator

        # run engine
        if self._output_dir is not None:
            logger.info(
                f"Running {self.__class__.__name__} engine with batch size [{self._dataloader.batch_size}] and output_dir: {self._output_dir}"
            )
        else:
            logger.info(f"Running engine {self.__class__.__name__}.")

        # load checkpoint if provided
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path=checkpoint_path)

        return self._engine.run(
            (
                FixedBatchIterator(self._dataloader, self._dataloader.batch_size)
                if self._use_fixed_batch_iterator
                else self._dataloader
            ),
            max_epochs=self._max_epochs,
            epoch_length=self._epoch_length,
        )
