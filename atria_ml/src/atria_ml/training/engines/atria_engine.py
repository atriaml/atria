"""
Atria Engine Base Module

This module defines the `AtriaEngine` class, which serves as the base class for all
engines in the Atria framework. It provides common functionality for managing training,
validation, and testing processes.

Classes:
    - AtriaEngine: Base class for managing engine processes.

Dependencies:
    - ignite: For distributed training and engine management.
    - atria_core.logger: For logging utilities.
    - atria_ml.training.configs: For logging configurations.
    - atria_ml.training.engines: For engine utilities and steps.

Author: Your Name (your.email@example.com)
Date: 2025-04-14
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

import copy
from abc import abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from atria_core.logger.logger import get_logger

from atria_ml.training.configs.logging_config import LoggingConfig
from atria_ml.training.engines.engine_steps.base import BaseEngineStep

if TYPE_CHECKING:
    import torch
    from atria_models.pipelines.atria_model_pipeline import AtriaModelPipeline
    from ignite.engine import Engine, Events, State
    from ignite.handlers import TensorboardLogger
    from torch.utils.data import DataLoader
    from webdataset import WebLoader

logger = get_logger(__name__)


class AtriaEngine:
    """
    AtriaEngine class for managing training, validation, and testing processes.

    Attributes:
        max_epochs (Optional[int]): Maximum number of epochs for training.
        epoch_length (Optional[int]): Number of iterations per epoch.
        outputs_to_running_avg (Optional[List[str]]): Outputs to compute running averages for.
        logging (LoggingConfig): Logging configuration.
        metric_logging_prefix (Optional[str]): Prefix for metric logging.
        sync_batchnorm (bool): Whether to synchronize batch normalization across devices.
        test_run (bool): Whether this is a test run.
        use_fixed_batch_iterator (bool): Whether to use a fixed batch iterator.
        checkpoints_dir (str): Directory for saving checkpoints.
    """

    def __init__(
        self,
        max_epochs: int | None = None,
        epoch_length: int | None = None,
        outputs_to_running_avg: list[str] | None = None,
        logging: LoggingConfig = LoggingConfig(),
        metric_logging_prefix: str | None = None,
        sync_batchnorm: bool = False,
        test_run: bool = False,
        use_fixed_batch_iterator: bool = False,
        checkpoints_dir: str = "checkpoints",
    ):
        """
        Initialize the AtriaEngine instance.

        Args:
            max_epochs (Optional[int]): Maximum number of epochs for training.
            epoch_length (Optional[int]): Number of iterations per epoch.
            outputs_to_running_avg (Optional[List[str]]): Outputs to compute running averages for.
            logging (LoggingConfig): Logging configuration.
            metric_logging_prefix (Optional[str]): Prefix for metric logging.
            sync_batchnorm (bool): Whether to synchronize batch normalization across devices.
            test_run (bool): Whether this is a test run.
            use_fixed_batch_iterator (bool): Whether to use a fixed batch iterator.
            checkpoints_dir (str): Directory for saving checkpoints.
        """
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
        self._checkpoints_dir = checkpoints_dir
        self._engine = None
        self._engine_step = None
        self._device = None
        self._tb_logger = None
        self._output_dir = None
        self._model_pipeline = None
        self._dataloader = None
        self._progress_bar = None
        self._event_handlers = []

    @property
    def batches_per_epoch(self) -> int:
        """
        Get the number of batches per epoch.

        Returns:
            int: Number of batches per epoch.
        """
        return len(self._dataloader)

    @property
    def steps_per_epoch(self) -> int:
        """
        Get the number of steps per epoch.

        Returns:
            int: Number of steps per epoch.
        """
        return self.batches_per_epoch

    @property
    def total_update_steps(self) -> int:
        """
        Get the total number of update steps.

        Returns:
            int: Total number of update steps.
        """
        return self.steps_per_epoch * self._max_epochs

    def add_event_handler(self, event: Events, handler: Callable):
        """
        Add an event handler to the engine.

        Args:
            event (Events): The event to attach the handler to.
            handler (Callable): The handler function.
        """
        self._event_handlers.append((event, handler))

    @abstractmethod
    def _setup_engine_step(self) -> BaseEngineStep:
        """
        Abstract method for engine step. Must be implemented by subclasses.
        """

    def build(
        self,
        output_dir: str | Path | None,
        model_pipeline: AtriaModelPipeline,
        dataloader: DataLoader | WebLoader,
        device: str | torch.device,
        tb_logger: TensorboardLogger | None = None,
    ) -> AtriaEngine:
        """
        Build the engine with the specified configurations.

        Args:
            output_dir (Optional[Union[str, Path]]): Directory for output files.
            model_pipeline (AtriaModelPipeline): Model pipeline to use.
            dataloader (Union[DataLoader, WebLoader]): Data loader for input data.
            device (Union[str, torch.device]): Device to run the engine on.
            tb_logger (Optional[TensorboardLogger]): Tensorboard logger for logging.

        Returns:
            AtriaEngine: The configured engine instance.
        """
        import torch
        from ignite.handlers import ProgressBar

        self._output_dir = output_dir
        self._model_pipeline = model_pipeline
        self._dataloader = dataloader
        self._device = torch.device(device)
        self._tb_logger = tb_logger

        # move task module models to device
        self._model_pipeline.to_device(self._device, sync_bn=self._sync_batchnorm)

        # initialize the engine step
        self._engine_step = self._setup_engine_step()

        # initialize the progress bar
        self._progress_bar = ProgressBar(
            desc=f"Stage [{self._engine_step.stage}]", persist=True
        )

        # attach the progress bar to task module
        self._model_pipeline.progress_bar = self._progress_bar

        # initialize the Ignite engine
        self._engine = self._initialize_ignite_engine()

        return self

    def run(self) -> State:
        """
        Run the engine.

        Returns:
            State: The state of the engine after execution.
        """
        from atria_ml.training.engines.utilities import FixedBatchIterator

        # run engine
        if self._output_dir is not None:
            logger.info(
                f"Running {self.__class__.__name__} engine with batch size [{self._dataloader.batch_size}] and output_dir: {self._output_dir}"
            )
        else:
            logger.info(f"Running engine {self.__class__.__name__}.")

        return self._engine.run(
            (
                FixedBatchIterator(self._dataloader, self._dataloader.batch_size)
                if self._use_fixed_batch_iterator
                else self._dataloader
            ),
            max_epochs=self._max_epochs,
            epoch_length=self._epoch_length,
        )

    def cleanup(self) -> None:
        """
        Cleanup resources used by the engine.
        """
        if self._engine is not None:
            self._engine_step = None
            self._engine = None

        if self._dataloader is not None:
            self._dataloader = None

        if self._model_pipeline is not None:
            self._model_pipeline._progress_bar = None
            self._model_pipeline._tb_logger = None
            self._model_pipeline = None

        if self._tb_logger is not None:
            self._tb_logger.close()
            self._tb_logger = None

        if self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None

    def _initialize_ignite_engine(self) -> Engine:
        """
        Initialize the engine.

        Returns:
            Engine: The initialized engine instance.
        """
        from ignite.engine import Engine

        engine = Engine(self._engine_step)
        engine.logger.propagate = False
        return self._configure_engine(engine)

    def _configure_engine(self, engine: Engine) -> Engine:
        """
        Configure the engine with handlers and settings.

        Args:
            engine (Engine): The engine instance to configure.
        """
        self._configure_test_run(engine=engine)
        self._configure_metrics(engine=engine)
        self._configure_running_avg_logging(engine=engine)
        self._configure_progress_bar(engine=engine)
        self._configure_tb_logger(engine=engine)
        self._attach_event_handlers(engine=engine)

        return engine

    def _configure_gpu_stats_callback(self, engine: Engine):
        """
        Configure GPU stats callback for the engine.

        Args:
            engine (Engine): The engine instance.
        """
        from atria_ml.training.engines.utilities import (
            _attach_gpu_stats_callback_to_engine,
        )

        if self._logging.log_gpu_stats:
            _attach_gpu_stats_callback_to_engine(engine, self._logging.logging_steps)

    def _configure_time_profiler(self, engine: Engine):
        """
        Configure time profiler for the engine.

        Args:
            engine (Engine): The engine instance.
        """
        from atria_ml.training.engines.utilities import _attach_time_profiler_to_engine

        if self._logging.profile_time:
            _attach_time_profiler_to_engine(engine)

    def _configure_metrics(self, engine: Engine) -> None:
        """
        Configure metrics for the engine.

        Args:
            engine (Engine): The engine instance.
        """
        from atria_ml.training.engines.utilities import _attach_metrics_to_engine

        if len(self._model_pipeline.metrics) > 0:
            logger.info(
                f"Attaching metrics {self._model_pipeline.metrics} to engine [{self.__class__.__name__}]"
            )
            _attach_metrics_to_engine(
                engine=engine,
                metrics=copy.deepcopy(self._model_pipeline.metrics),
                prefix=self._metric_logging_prefix,
                stage=self._engine_step.stage,
            )

    def _configure_running_avg_logging(self, engine: Engine) -> None:
        """
        Configure running average logging for the engine.

        Args:
            engine (Engine): The engine instance.
        """
        from atria_ml.training.engines.utilities import _attach_output_logging_to_engine

        _attach_output_logging_to_engine(
            engine=engine,
            stage=self._engine_step.stage,
            outputs_to_running_avg=self._outputs_to_running_avg,
        )

    def _configure_progress_bar(self, engine: Engine) -> None:
        """
        Configure progress bar for the engine.

        Args:
            engine (Engine): The engine instance.
        """
        import ignite.distributed as idist
        from ignite.engine import Events

        from atria_ml.training.engines.utilities import _log_eval_metrics

        if idist.get_rank() == 0:
            self._progress_bar.attach(
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
                    tag=self._engine_step.stage,
                    metrics=engine.state.metrics,
                )

            @engine.on(Events.TERMINATE | Events.INTERRUPT)
            def progress_on_terminate(engine: Engine) -> None:
                logger.info(
                    f"Engine [{self.__class__.__name__}] terminated after {engine.state.epoch} epochs."
                )
                self._progress_bar.close()

    def _configure_tb_logger(self, engine: Engine):
        """
        Configure Tensorboard logger for the engine.

        Args:
            engine (Engine): The engine instance.
        """
        import ignite.distributed as idist
        from ignite.engine import Events

        if (
            idist.get_rank() == 0
            and self._tb_logger is not None
            and self._logging.log_to_tb
        ):
            # attach tb logger to validation engine
            self._tb_logger.attach_output_handler(
                engine,
                event_name=Events.EPOCH_COMPLETED,
                metric_names="all",
                tag="epoch",
            )

    def _configure_test_run(self, engine: Engine):
        """
        Configure test run settings for the engine.

        Args:
            engine (Engine): The engine instance.
        """
        from ignite.engine import Events

        if self._test_run:
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
            engine.add_event_handler(
                Events.ITERATION_STARTED, print_iteration_started_info
            )
            engine.add_event_handler(
                Events.ITERATION_COMPLETED, print_iteration_completed_info
            )

    def _attach_event_handlers(self, engine: Engine):
        """
        Attach custom event handlers to the engine.

        Args:
            engine (Engine): The engine instance.
        """
        for event, handler in self._event_handlers:
            engine.add_event_handler(event, handler)
