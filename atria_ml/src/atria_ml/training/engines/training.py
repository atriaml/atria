"""
Training Engine Module

This module defines the `TrainingEngine` class, which is responsible for managing the
training process in the Atria framework. It includes functionality for configuring
optimizers, learning rate schedulers, early stopping, model checkpointing, and more.

Classes:
    - TrainingEngine: A class for managing the training process.

Dependencies:
    - ignite: For distributed training and engine management.
    - atria_core.logger: For logging utilities.
    - atria_ml.registry: For registering the training engine.
    - atria_ml.training.configs: For various training configurations.
    - atria_ml.training.engines: For engine utilities and steps.
    - atria_ml.schedulers: For learning rate scheduler types.

Author: Your Name (your.email@example.com)
Date: 2025-04-14
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

import copy
import math
from collections import OrderedDict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from atria_core.logger.logger import get_logger

from atria_ml.registry import ENGINE
from atria_ml.registry.registry_groups import LRSchedulerBuilder, OptimizerBuilder
from atria_ml.training.configs.early_stopping_config import EarlyStoppingConfig
from atria_ml.training.configs.gradient_config import GradientConfig
from atria_ml.training.configs.logging_config import LoggingConfig
from atria_ml.training.configs.model_checkpoint import ModelCheckpointConfig
from atria_ml.training.configs.model_ema_config import ModelEmaConfig
from atria_ml.training.configs.warmup_config import WarmupConfig
from atria_ml.training.engines.atria_engine import AtriaEngine
from atria_ml.training.engines.engine_steps.training import TrainingStep
from atria_ml.training.engines.evaluation import ValidationEngine, VisualizationEngine
from atria_ml.training.engines.utilities import RunConfig

if TYPE_CHECKING:
    import torch
    from atria_models.pipelines.atria_model_pipeline import AtriaModelPipeline
    from ignite.engine import Engine, State
    from ignite.handlers import TensorboardLogger
    from torch.utils.data import DataLoader
    from webdataset import WebLoader

logger = get_logger(__name__)


@ENGINE.register(
    "default_training_engine",
    defaults=[
        "_self_",
        {"/optimizer@optimizer": "adam"},
        {"/lr_scheduler@lr_scheduler": "cosine_annealing_lr"},
    ],
)
class TrainingEngine(AtriaEngine):
    """
    TrainingEngine Class

    This class is responsible for managing the training process in the Atria framework.
    It provides methods for configuring optimizers, learning rate schedulers, early stopping,
    model checkpointing, and other training-related functionalities.

    Attributes:
        _optimizer (OptimizerType): Factory for creating optimizers.
        _lr_scheduler (Optional[LRSchedulerType]): Factory for creating learning rate schedulers.
        _eval_training (Optional[bool]): Flag to enable evaluation during training.
        _stop_on_nan (bool): Flag to stop training if NaN values are encountered.
        _clear_cuda_cache (Optional[bool]): Flag to clear CUDA cache during training.
        _model_ema_config (Optional[ModelEmaConfig]): Configuration for model EMA.
        _warmup_config (WarmupConfig): Configuration for warmup steps.
        _early_stopping (EarlyStoppingConfig): Configuration for early stopping.
        _model_checkpoint_config (ModelCheckpointConfig): Configuration for model checkpointing.
        _gradient_config (GradientConfig): Configuration for gradient accumulation.
        _ema_handler (Optional[Any]): Handler for model EMA.

    Methods:
        __init__: Initializes the TrainingEngine instance.
        steps_per_epoch: Returns the number of steps per epoch.
        total_warmup_steps: Returns the total number of warmup steps.
        _initialize_engine: Initializes the Ignite engine.
        build: Builds the training engine with the provided configurations.
        _configure_train_sampler: Configures the training sampler for distributed training.
        _configure_nan_callback: Configures the callback to handle NaN values.
        _configure_cuda_cache_callback: Configures the callback to clear CUDA cache.
        _configure_model_ema_callback: Configures the model EMA callback.
        _configure_metrics: Configures metrics for evaluation during training.
        _configure_schedulers: Configures learning rate schedulers.
        _configure_progress_bar: Configures the progress bar for training.
        _configure_tb_logger: Configures the TensorBoard logger.
        _prepare_checkpoint_state_dict: Prepares the state dictionary for checkpointing.
        _configure_model_checkpointer: Configures the model checkpointer.
        _load_training_state_from_checkpoint: Loads the training state from a checkpoint.
        _configure_early_stopping_callback: Configures the early stopping callback.
        _configure_validation_engine: Configures the validation engine.
        _configure_visualization_engine: Configures the visualization engine.
        _register_events: Registers custom events for the engine.
        _configure_engine: Configures the training engine.
        _print_configuration_info: Prints the configuration information of the training engine.
        run: Runs the training engine.
    """

    def __init__(
        self,
        optimizer: OptimizerBuilder,
        lr_scheduler: LRSchedulerBuilder | None = None,
        max_epochs: int = 100,
        epoch_length: int | None = None,
        outputs_to_running_avg: list[str] | None = None,
        logging: LoggingConfig = LoggingConfig(),
        metric_logging_prefix: str | None = None,
        sync_batchnorm: bool = False,
        test_run: bool = False,
        use_fixed_batch_iterator: bool = False,
        eval_training: bool | None = False,
        stop_on_nan: bool = True,
        clear_cuda_cache: bool | None = True,
        model_ema_config: ModelEmaConfig | None = ModelEmaConfig(),
        warmup_config: WarmupConfig = WarmupConfig(),
        early_stopping: EarlyStoppingConfig = EarlyStoppingConfig(),
        model_checkpoint_config: ModelCheckpointConfig = ModelCheckpointConfig(),
        gradient_config: GradientConfig = GradientConfig(),
        with_amp: bool = False,
    ):
        """
        Initializes the TrainingEngine instance.

        Args:
            engine_step (TrainingStep): The training step to be executed.
            optimizer (OptimizerType): Factory for creating optimizers.
            lr_scheduler (Optional[LRSchedulerType]): Factory for creating learning rate schedulers.
            max_epochs (int): Maximum number of epochs for training.
            epoch_length (Optional[int]): Length of each epoch.
            outputs_to_running_avg (Optional[List[str]]): Outputs to compute running averages.
            logging (LoggingConfig): Configuration for logging.
            metric_logging_prefix (Optional[str]): Prefix for metric logging.
            sync_batchnorm (bool): Flag to synchronize batch normalization across devices.
            test_run (bool): Flag to enable test run mode.
            use_fixed_batch_iterator (bool): Flag to use a fixed batch iterator.
            eval_training (Optional[bool]): Flag to enable evaluation during training.
            stop_on_nan (bool): Flag to stop training if NaN values are encountered.
            clear_cuda_cache (Optional[bool]): Flag to clear CUDA cache during training.
            model_ema_config (Optional[ModelEmaConfig]): Configuration for model EMA.
            warmup_config (WarmupConfig): Configuration for warmup steps.
            early_stopping (EarlyStoppingConfig): Configuration for early stopping.
            model_checkpoint_config (ModelCheckpointConfig): Configuration for model checkpointing.
            gradient_config (GradientConfig): Configuration for gradient accumulation.
            grad_scaler (Optional[torch.cuda.amp.GradScaler]): GradScaler for mixed precision training.
            with_amp (bool): Flag to enable automatic mixed precision (AMP) training.
        """
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._eval_training = eval_training
        self._stop_on_nan = stop_on_nan
        self._clear_cuda_cache = clear_cuda_cache
        self._model_ema_config = model_ema_config
        self._warmup_config = warmup_config
        self._early_stopping = early_stopping
        self._model_checkpoint_config = model_checkpoint_config
        self._gradient_config = gradient_config
        self._with_amp = with_amp
        self._ema_handler = None
        self._optimizers = None
        self._lr_schedulers = None

        super().__init__(
            max_epochs=max_epochs,
            epoch_length=epoch_length,
            outputs_to_running_avg=outputs_to_running_avg,
            logging=logging,
            metric_logging_prefix=metric_logging_prefix,
            sync_batchnorm=sync_batchnorm,
            test_run=test_run,
            use_fixed_batch_iterator=use_fixed_batch_iterator,
            checkpoints_dir=model_checkpoint_config.dir,
        )

    @property
    def steps_per_epoch(self) -> int:
        """
        Returns the number of steps per epoch.

        Returns:
            int: Number of steps per epoch.
        """
        return (
            self.batches_per_epoch // self._gradient_config.gradient_accumulation_steps
        )

    @property
    def total_warmup_steps(self):
        """
        Returns the total number of warmup steps.

        Returns:
            int: Total number of warmup steps.
        """
        if self._warmup_config.warmup_steps is None:
            self._warmup_config.warmup_steps = 0
        if self._warmup_config.warmup_ratio is None:
            self._warmup_config.warmup_ratio = 0.0
        return (
            self._warmup_config.warmup_steps
            if self._warmup_config.warmup_steps > 0
            else math.ceil(self.total_update_steps * self._warmup_config.warmup_ratio)
        )

    def _setup_engine_step(self):
        return TrainingStep(
            model_pipeline=self._model_pipeline,
            device=self._device,
            optimizers=self._optimizers,
            gradient_config=self._gradient_config,
            grad_scaler=self._grad_scaler,
            with_amp=self._with_amp,
            test_run=self._test_run,
        )

    def build(
        self,
        run_config: RunConfig,
        output_dir: str | Path | None,
        model_pipeline: AtriaModelPipeline,
        dataloader: DataLoader | WebLoader,
        device: str | torch.device,
        grad_scaler: torch.cuda.amp.GradScaler | None = None,
        validation_engine: ValidationEngine = None,
        visualization_engine: VisualizationEngine = None,
        tb_logger: TensorboardLogger | None = None,
    ) -> TrainingEngine:
        """
        Builds the training engine with the provided configurations.

        Args:
            run_config (RunConfig): Configuration for the training run.
            output_dir (str | Path | None): Directory for saving outputs.
            model_pipeline (AtriaModelPipeline): Model pipeline for training.
            dataloader (Union[DataLoader, WebLoader]): Data loader for training data.
            device (Union[str, torch.device]): Device for training.
            validation_engine (ValidationEngine, optional): Validation engine for evaluation.
            visualization_engine (VisualizationEngine, optional): Visualization engine for visualizing results.
            tb_logger (TensorboardLogger, optional): TensorBoard logger for logging metrics.

        Returns:
            TrainingEngine: The configured training engine.
        """

        import ignite.distributed as idist
        import torch
        from atria_core.constants import _DEFAULT_OPTIMIZER_PARAMETERS_KEY
        from ignite.handlers import ProgressBar

        from atria_ml.training.engines.utilities import (
            _print_optimizers_info,
            _print_schedulers_info,
        )

        self._run_config = run_config
        self._output_dir = output_dir
        self._model_pipeline = model_pipeline
        self._dataloader = dataloader
        self._device = torch.device(device)
        self._grad_scaler = grad_scaler
        self._tb_logger = tb_logger
        self._validation_engine = validation_engine
        self._visualization_engine = visualization_engine

        # convert optimizers to dict if not already
        if not isinstance(self._optimizer, dict):
            self._optimizer = {_DEFAULT_OPTIMIZER_PARAMETERS_KEY: self._optimizer}

        optimized_parameters_dict = self._model_pipeline.optimizer_parameters()
        assert len(optimized_parameters_dict) == len(self._optimizer), (
            "Number of optimizers must match the number of model parameter groups defined in the task_module. "
            f"Optimizers: {len(self._optimizer)} != Model parameter groups: {len(optimized_parameters_dict)}"
        )

        self._optimizers = {}
        for k, opt in self._optimizer.items():
            if k not in optimized_parameters_dict.keys():
                raise ValueError(
                    f"Your optimizer configuration does not align with the model optimizer "
                    f"parameter groups. {k} =/= {optimized_parameters_dict.keys()}"
                )

            # initialize the optimizers from partial with the model parameters
            self._optimizers[k] = idist.auto_optim(
                opt(params=optimized_parameters_dict[k])
            )

        # print information
        _print_optimizers_info(self._optimizers)

        # initialize lr schedulers partials
        self._lr_schedulers = {}
        if self._lr_scheduler is not None:
            # convert lr_schedulers to dict if not already
            if not isinstance(self._lr_scheduler, dict):
                self._lr_scheduler = {
                    _DEFAULT_OPTIMIZER_PARAMETERS_KEY: self._lr_scheduler
                }

            for k, sch in self._lr_scheduler.items():
                runtime_kwargs = {}
                for kwarg in [
                    "total_update_steps",
                    "total_warmup_steps",
                    "steps_per_epoch",
                ]:
                    if kwarg in sch.get_possible_args():
                        runtime_kwargs[kwarg] = getattr(self, kwarg)

                logger.debug(
                    f"Initializing lr scheduler {sch} with runtime kwargs: {runtime_kwargs}"
                )
                self._lr_schedulers[k] = sch(
                    optimizer=self._optimizers[k], **runtime_kwargs
                )

            # print information
            _print_schedulers_info(self._lr_schedulers)

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

    def run(self, resume_checkpoint: str | None) -> State:
        """
        Runs the training engine.

        Returns:
            State: The state of the training engine after execution.
        """
        from atria_ml.training.engines.utilities import FixedBatchIterator

        # load training state from checkpoint
        self._load_training_state_from_checkpoint(
            self._engine, resume_checkpoint=resume_checkpoint
        )

        resume_epoch = self._engine.state.epoch
        if (
            self._engine._is_done(self._engine.state)
            and resume_epoch >= self._max_epochs
        ):  # if we are resuming from last checkpoint and training is already finished
            logger.warning(
                "Training has already been finished! Either increase the number of "
                f"epochs (current={self._max_epochs}) >= {resume_epoch} "
                "OR reset the training from start."
            )
            return

        # run engine
        logger.info(
            f"Running training with batch size [{self._dataloader.batch_size}] and output_dir:\n\t{self._output_dir}"
        )
        return self._engine.run(
            (
                FixedBatchIterator(self._dataloader, self._dataloader.batch_size)
                if self._use_fixed_batch_iterator
                else self._dataloader
            ),
            max_epochs=self._max_epochs,
            epoch_length=self._epoch_length,
        )

    def _initialize_ignite_engine(self) -> Engine:
        """
        Initializes the Ignite engine.

        Returns:
            Engine: The initialized Ignite engine.
        """
        from ignite.engine import Engine

        class IgniteTrainingEngine(Engine):
            def state_dict(self) -> OrderedDict:
                """
                Returns the state dictionary of the engine.

                Returns:
                    OrderedDict: State dictionary of the engine.
                """
                state_dict = super().state_dict()
                if hasattr(self.state, "optimizer_step"):
                    state_dict["optimizer_step"] = self.state.optimizer_step
                return state_dict

            def load_state_dict(self, state_dict: Mapping) -> None:
                """
                Loads the state dictionary into the engine.

                Args:
                    state_dict (Mapping): State dictionary to load.
                """
                super().load_state_dict(state_dict)
                if hasattr(self.state, "optimizer_step"):
                    self.state.optimizer_step = state_dict["optimizer_step"]

        engine = IgniteTrainingEngine(self._engine_step)
        engine.logger.propagate = False
        return self._configure_engine(engine)

    def _configure_train_sampler(self, engine: Engine):
        """
        Configures the training sampler for distributed training.

        Args:
            engine (Engine): The training engine.
        """
        import ignite.distributed as idist
        from torch.utils.data.distributed import DistributedSampler

        if idist.get_world_size() > 1:
            from ignite.engine import Events

            train_sampler = self._dataloader.sampler
            if not isinstance(train_sampler, DistributedSampler):
                raise TypeError(
                    "Train sampler should be torch DistributedSampler and have `set_epoch` method"
                )

            @engine.on(Events.EPOCH_STARTED)
            def distrib_set_epoch(engine: Engine) -> None:
                cast(DistributedSampler, train_sampler).set_epoch(
                    engine.state.epoch - 1
                )

        else:
            # check whether the correct training sample is being used
            if self._dataloader.sampler is not None and isinstance(
                self._dataloader.sampler, DistributedSampler
            ):
                logger.warning(
                    "Argument train_sampler is a distributed sampler,"
                    " but either there is no distributed setting or world size is < 2. "
                    "Train sampler argument will be ignored",
                    UserWarning,
                )

    def _configure_nan_callback(self, engine: Engine):
        """
        Configures the callback to handle NaN values.

        Args:
            engine (Engine): The training engine.
        """
        from atria_ml.training.engines.utilities import _attach_nan_callback_to_engine

        if self._stop_on_nan:
            _attach_nan_callback_to_engine(engine)

    def _configure_cuda_cache_callback(self, engine: Engine):
        """
        Configures the callback to clear CUDA cache.

        Args:
            engine (Engine): The training engine.
        """
        from atria_ml.training.engines.utilities import (
            _attach_cuda_cache_callback_to_engine,
        )

        if self._clear_cuda_cache:
            _attach_cuda_cache_callback_to_engine(engine)

    def _configure_model_ema_callback(self, engine: Engine) -> None:
        """
        Configures the model EMA callback.

        Args:
            engine (Engine): The training engine.
        """
        from atria_models.utilities.ddp_model_proxy import ModuleProxyWrapper
        from atria_models.utilities.nn_modules import AtriaModelDict
        from torchinfo import summary

        from atria_ml.training.engines.events import OptimizerEvents
        from atria_ml.training.handlers.ema_handler import AtriaEMAHandler

        if self._model_ema_config.enabled:
            trainable_model = (
                self._model_pipeline.model.trainable_models
                if isinstance(self._model_pipeline.model, AtriaModelDict)
                else self._model_pipeline.model
            )
            if isinstance(trainable_model, ModuleProxyWrapper):
                trainable_model = trainable_model.module

            self._ema_handler = AtriaEMAHandler(
                trainable_model,
                momentum=self._model_ema_config.momentum,
                momentum_warmup=self._model_ema_config.momentum_warmup,
                warmup_iters=self._model_ema_config.warmup_iters,
                handle_buffers="update",
            )

            logger.info(
                f"Attaching EMAHandler with following configuration: {self._model_ema_config}"
            )
            logger.info("Ema Model:")
            logger.info(summary(self._ema_handler.ema_model, verbose=0, depth=2))
            self._ema_handler.attach(
                engine,
                name="ema_momentum",
                event=OptimizerEvents.optimizer_step(
                    every=self._model_ema_config.update_every
                ),
            )

    def _configure_metrics(self, engine: Engine) -> None:
        """
        Configures metrics for evaluation during training.

        Args:
            engine (Engine): The training engine.
        """
        if self._eval_training:
            super()._configure_metrics(engine)

    def _configure_schedulers(self, engine: Engine) -> None:
        """
        Configures learning rate schedulers.

        Args:
            engine (Engine): The training engine.
        """
        from ignite.engine import Events
        from ignite.handlers import (
            LRScheduler,
            ParamScheduler,
            ReduceLROnPlateauScheduler,
            create_lr_scheduler_with_warmup,
        )
        from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, StepLR

        from atria_ml.training.engines.events import OptimizerEvents

        if self._lr_schedulers is None:
            return

        for k, inner_sch in self._lr_schedulers.items():
            if inner_sch is None:
                continue

            if self.total_warmup_steps > 0:
                logger.info(
                    f"Initialized lr scheduler {inner_sch.__class__.__name__} with warmup. "
                )
                logger.info(f"Warmup ratio = {self._warmup_config.warmup_ratio}. ")
                logger.info(
                    f"Number of warmup steps = {self.total_warmup_steps}. This corresponds to optimizer updates, "
                    "not total batches in epoch and therefore its scaled by grad "
                    f"acummulation steps = {self._gradient_config.gradient_accumulation_steps}."
                )

                if isinstance(inner_sch, StepLR | MultiStepLR):
                    logger.info(
                        "Warmup updates are triggered per optimizer steps whereas the scheduler updates are triggered per epoch."
                    )
                    sch = create_lr_scheduler_with_warmup(
                        inner_sch,
                        warmup_start_value=0.0,
                        warmup_duration=self.total_warmup_steps,
                    )

                    # we want warmup on optimizer update steps and step_lr on epochs, so we create two events first for steps
                    # and then for epochs
                    # Trigger scheduler on iteration_started events before reaching warmup_duration
                    combined_events = OptimizerEvents.optimizer_step(
                        event_filter=lambda _, __: engine.state.optimizer_step
                        <= self.total_warmup_steps
                    )

                    # Trigger scheduler on epoch_started events after the warm-up. Epochs are 1-based, thus we do 1 +
                    combined_events |= Events.EPOCH_STARTED(
                        event_filter=lambda _, __: engine.state.epoch
                        > 1 + self.total_warmup_steps / self.steps_per_epoch
                    )

                    engine.add_event_handler(combined_events, sch)

                    # update scheduler in dict
                    self._lr_schedulers[k] = sch
                elif isinstance(inner_sch, ReduceLROnPlateauScheduler):
                    logger.info(
                        "Warmup updates are triggered per optimizer steps whereas the scheduler updates are triggered per validation step."
                    )
                    # we want warmup on steps and step_lr on epochs, so we create two events first for steps
                    # and then for epochs
                    sch = create_lr_scheduler_with_warmup(
                        inner_sch,
                        warmup_start_value=0.0,
                        warmup_duration=self.total_warmup_steps,
                    )
                    engine.add_event_handler(
                        OptimizerEvents.optimizer_step(
                            event_filter=lambda _, __: engine.state.optimizer_step
                            <= self.total_warmup_steps
                        ),
                        sch.schedulers[0],
                    )

                    # Trigger scheduler on epoch_started events after the warm-up. Epochs are 1-based, thus we do 1 +
                    combined_events = Events.COMPLETED | Events.COMPLETED(
                        event_filter=lambda _, __: engine.state.epoch
                        > 1 + self.total_warmup_steps / self.steps_per_epoch
                    )

                    if self._validation_engine is not None:
                        self._validation_engine.add_event_handler(
                            combined_events, inner_sch
                        )
                    else:
                        logger.warning(
                            "ReduceLROnPlateauScheduler metric is initialized with no validation engine attached. "
                        )
                    self._lr_schedulers[k] = sch
                else:
                    logger.info(
                        "Both warmup updates and the scheduler updates are triggered per optimizer step."
                    )
                    sch = create_lr_scheduler_with_warmup(
                        inner_sch,
                        warmup_start_value=0.0,
                        warmup_duration=self.total_warmup_steps,
                    )
                    engine.add_event_handler(OptimizerEvents.optimizer_step, sch)

                    # update scheduler in dict
                    self._lr_schedulers[k] = sch
            else:
                if not isinstance(inner_sch, ParamScheduler):
                    # convert scheduler to ignite scheduler
                    sch = LRScheduler(inner_sch)
                else:
                    sch = inner_sch

                # update scheduler in dict
                if isinstance(inner_sch, StepLR | MultiStepLR | ExponentialLR):
                    logger.info(
                        f"Initialized lr scheduler {inner_sch.__class__.__name__}. Scheduler updates are triggered per epoch. "
                    )
                    engine.add_event_handler(Events.EPOCH_STARTED, sch)
                elif isinstance(inner_sch, ReduceLROnPlateauScheduler):
                    logger.info(
                        f"Initialized lr scheduler {inner_sch.__class__.__name__}. Scheduler updates are triggered per validation step. "
                    )
                    # inner_sch.trainer = training_engine
                    engine.add_event_handler(Events.COMPLETED, sch)
                else:
                    logger.info(
                        f"Initialized lr scheduler {inner_sch.__class__.__name__}. Scheduler updates are triggered per optimizer step. "
                    )
                    engine.add_event_handler(OptimizerEvents.optimizer_step, sch)
                self._lr_schedulers[k] = sch

    def _configure_progress_bar(self, engine: Engine) -> None:
        """
        Configures the progress bar for training.

        Args:
            engine (Engine): The training engine.
        """
        from ignite.engine import Events

        from atria_ml.training.engines.utilities import _log_training_metrics
        from atria_ml.training.utilities.constants import TrainingStage

        self._progress_bar.attach(
            engine,
            metric_names="all",
            event_name=Events.ITERATION_COMPLETED(every=self._logging.refresh_rate),
            state_attributes=["optimizer_step", "ema_momentum"],
        )

        @engine.on(Events.EPOCH_COMPLETED)
        def progress_on_epoch_completed(engine: Engine) -> None:
            metrics = copy.deepcopy(engine.state.metrics)

            if hasattr(engine.state, "ema_momentum"):
                metrics["ema/mom"] = engine.state.ema_momentum

            _log_training_metrics(
                logger=logger,
                epoch=engine.state.epoch,
                elapsed=engine.state.times["EPOCH_COMPLETED"],
                tag=TrainingStage.train,
                metrics=metrics,
            )

        @engine.on(Events.EXCEPTION_RAISED)
        def progress_on_terminate(exception: Exception) -> None:
            self._tb_logger.close()
            self._progress_bar.close()

    def _configure_tb_logger(self, engine: Engine):
        """
        Configures the TensorBoard logger.

        Args:
            engine (Engine): The training engine.
        """
        import ignite.distributed as idist

        if (
            idist.get_rank() == 0
            and self._tb_logger is not None
            and self._logging.log_to_tb
        ):
            from ignite.engine import Events

            # attach handler to plot trainer's loss every 'logging_steps' iterations
            self._tb_logger.attach_output_handler(
                engine,
                event_name=Events.ITERATION_COMPLETED(
                    every=self._logging.logging_steps
                ),
                tag="step",
                metric_names="all",
            )

            # attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at every
            # 'logging_steps' iteration
            for param_name in ["lr", "weight_decay"]:
                for k, opt in self._optimizers.items():
                    self._tb_logger.attach_opt_params_handler(
                        engine,
                        event_name=Events.ITERATION_STARTED(
                            every=self._logging.logging_steps
                        ),
                        optimizer=opt,
                        param_name=param_name,
                        tag=f"step/opt/{k}",
                    )

    def _prepare_checkpoint_state_dict(
        self, engine: Engine, save_weights_only: bool = False
    ) -> dict[str, Any]:
        """
        Prepares the state dictionary for checkpointing.

        Args:
            engine (Engine): The training engine.
            save_weights_only (bool): Flag to save only weights in the checkpoint.

        Returns:
            Dict[str, Any]: State dictionary for checkpointing.
        """
        from atria_ml.training.engines.utilities import (
            MODEL_PIPELINE_CHECKPOINT_KEY,
            RUN_CONFIG_KEY,
            TRAINING_ENGINE_KEY,
        )

        checkpoint_state_dict = {
            RUN_CONFIG_KEY: self._run_config,
            TRAINING_ENGINE_KEY: engine,
            MODEL_PIPELINE_CHECKPOINT_KEY: self._model_pipeline,
        }

        # add optimizers and lr/wd scheduler states to checkpoint_state_dict
        lr_schedulers_checkpoint_state_dict = (
            {f"lr_sch_{k}": v for k, v in self._lr_schedulers.items()}
            if self._lr_schedulers
            else {}
        )
        checkpoint_state_dict = {
            **checkpoint_state_dict,
            **{f"opt_{k}": v for k, v in self._optimizers.items()},
            **lr_schedulers_checkpoint_state_dict,
        }

        # add ema handler state to checkpoint_state_dict
        if self._ema_handler is not None:
            checkpoint_state_dict["ema_model"] = self._ema_handler.ema_model
            checkpoint_state_dict["ema_momentum_scheduler"] = (
                self._ema_handler.momentum_scheduler
            )

        # if only to save weights, remove all other keys
        if save_weights_only:
            for k in list(checkpoint_state_dict.keys()):
                if k not in [TRAINING_ENGINE_KEY, MODEL_PIPELINE_CHECKPOINT_KEY]:
                    checkpoint_state_dict.pop(k)

        return checkpoint_state_dict

    def _configure_model_checkpointer(self, engine: Engine):
        """
        Configures the model checkpointer.

        Args:
            engine (Engine): The training engine.
        """
        from ignite.engine import Events
        from ignite.handlers import DiskSaver
        from ignite.handlers.checkpoint import BaseSaveHandler, Checkpoint

        # setup checkpoint saving if required
        if self._model_checkpoint_config:
            logger.info("Configuring model checkpointing with the following config:")
            logger.info(f"{self._model_checkpoint_config}")
            checkpoint_state_dict = self._prepare_checkpoint_state_dict(
                engine,
                save_weights_only=self._model_checkpoint_config.save_weights_only,
            )

            checkpoint_dir = Path(self._output_dir) / self._model_checkpoint_config.dir
            save_handler = DiskSaver(checkpoint_dir, require_empty=False)
            if self._model_checkpoint_config.save_per_epoch:
                checkpoint_handler = Checkpoint(
                    checkpoint_state_dict,
                    cast(Callable | BaseSaveHandler, save_handler),
                    filename_prefix=self._model_checkpoint_config.name_prefix,
                    global_step_transform=lambda *_: engine.state.epoch,
                    n_saved=self._model_checkpoint_config.n_saved,
                    include_self=True,
                )
                engine.add_event_handler(
                    Events.EPOCH_COMPLETED(
                        every=self._model_checkpoint_config.save_every_iters
                    ),
                    checkpoint_handler,
                )
            else:
                checkpoint_handler = Checkpoint(
                    checkpoint_state_dict,
                    cast(Callable | BaseSaveHandler, save_handler),
                    filename_prefix=self._model_checkpoint_config.name_prefix,
                    n_saved=self._model_checkpoint_config.n_saved,
                    include_self=True,
                )
                engine.add_event_handler(
                    Events.ITERATION_COMPLETED(
                        every=self._model_checkpoint_config.save_every_iters
                    )
                    | Events.COMPLETED,
                    checkpoint_handler,
                )

        if (
            self._validation_engine is not None
            and self._model_checkpoint_config.monitored_metric is not None
        ):
            from ignite.contrib.handlers import global_step_from_engine

            checkpoint_state_dict = self._prepare_checkpoint_state_dict(
                engine,
                save_weights_only=self._model_checkpoint_config.save_weights_only,
            )

            checkpoint_dir = Path(self._output_dir) / self._model_checkpoint_config.dir
            save_handler = DiskSaver(checkpoint_dir, require_empty=False)

            logger.info(
                f"Configuring best model checkpointing with monitored metric:\n\t{self._model_checkpoint_config.monitored_metric}"
            )
            best_model_saver = Checkpoint(
                checkpoint_state_dict,
                save_handler=DiskSaver(checkpoint_dir, require_empty=False),
                filename_prefix="best",
                n_saved=self._model_checkpoint_config.n_best_saved,
                global_step_transform=global_step_from_engine(engine),
                score_name=self._model_checkpoint_config.monitored_metric.replace(
                    "/", "-"
                ),
                score_function=Checkpoint.get_default_score_fn(
                    self._model_checkpoint_config.monitored_metric,
                    -1 if self._model_checkpoint_config.mode == "min" else 1.0,
                ),
                include_self=True,
            )
            self._validation_engine.add_event_handler(
                Events.COMPLETED, best_model_saver
            )

    def _load_training_state_from_checkpoint(
        self, engine: Engine, resume_checkpoint: str | None = None
    ):
        """
        Loads the training state from a checkpoint.

        Args:
            engine (Engine): The training engine.
        """
        from ignite.handlers.checkpoint import Checkpoint

        from atria_ml.training.engines.utilities import (
            MODEL_PIPELINE_CHECKPOINT_KEY,
            RUN_CONFIG_KEY,
        )

        if self._model_checkpoint_config.resume_from_checkpoint:
            import torch

            from atria_ml.training.utilities.checkpoints import find_resume_checkpoint

            checkpoint_state_dict = self._prepare_checkpoint_state_dict(
                engine,
                save_weights_only=self._model_checkpoint_config.save_weights_only,
            )
            checkpoint_dir = Path(self._output_dir) / self._model_checkpoint_config.dir
            resume_checkpoint = resume_checkpoint or find_resume_checkpoint(
                checkpoint_dir
            )
            if resume_checkpoint is not None:
                logger.info(
                    f"Checkpoint detected, resuming training from {resume_checkpoint}. "
                )
                logger.info(f"\t{resume_checkpoint}")
                resume_checkpoint = torch.load(resume_checkpoint, map_location="cpu")

                if RUN_CONFIG_KEY in resume_checkpoint:
                    self._run_config.compare_configs(resume_checkpoint[RUN_CONFIG_KEY])

                for k in list(checkpoint_state_dict.keys()):
                    if k not in list(resume_checkpoint.keys()):
                        logger.warning(
                            f"Object {k} not found in the resume checkpoint_state_dict."
                        )
                        del checkpoint_state_dict[k]

                load_state_dict = {**checkpoint_state_dict}
                if self._model_checkpoint_config.load_weights_only:
                    for k in list(checkpoint_state_dict.keys()):
                        if k not in [MODEL_PIPELINE_CHECKPOINT_KEY]:
                            load_state_dict.pop(k)

                Checkpoint.load_objects(
                    to_load=load_state_dict, checkpoint=resume_checkpoint, strict=False
                )

    def _configure_early_stopping_callback(self, engine: Engine) -> None:
        """
        Configures the early stopping callback.

        Args:
            engine (Engine): The training engine.
        """
        if self._early_stopping.enabled:
            from ignite.engine import Events
            from ignite.handlers import Checkpoint, EarlyStopping

            if self._validation_engine is None:
                raise ValueError(
                    "Validation engine is not attached to training. Early stopping can not be configured. "
                    "Did you set do_validation=True in the trainer?"
                )

            es_handler = EarlyStopping(
                patience=self._early_stopping.patience,
                score_function=Checkpoint.get_default_score_fn(
                    self._early_stopping.monitored_metric,
                    -1 if self._early_stopping.mode == "min" else 1.0,
                ),
                trainer=engine,
            )
            self._validation_engine.add_event_handler(Events.COMPLETED, es_handler)

    def _configure_validation_engine(self, engine: Engine) -> None:
        """
        Configures the validation engine.

        Args:
            engine (Engine): The training engine.
        """
        if self._validation_engine is not None:
            self._validation_engine.attach_to_engine(
                parent_engine=engine,
                steps_per_epoch=self.steps_per_epoch,
                ema_handler=self._ema_handler,
            )

    def _configure_visualization_engine(self, engine: Engine) -> None:
        """
        Configures the visualization engine.

        Args:
            engine (Engine): The training engine.
        """
        if self._visualization_engine is not None:
            self._visualization_engine.attach_to_engine(
                parent_engine=engine,
                steps_per_epoch=self.steps_per_epoch,
                ema_handler=self._ema_handler,
            )

    def _register_events(self, engine: Engine) -> None:
        """
        Registers custom events for the engine.

        Args:
            engine (Engine): The training engine.
        """
        from atria_ml.training.engines.events import OptimizerEvents

        engine.register_events(
            *OptimizerEvents,
            event_to_attr={
                OptimizerEvents.optimizer_step: OptimizerEvents.optimizer_step.value
            },
        )

    def _configure_engine(self, engine: Engine) -> Engine:
        """
        Configures the training engine.

        Args:
            engine (Engine): The training engine.
        """
        # register events if needed
        self._configure_test_run(engine=engine)
        self._register_events(engine=engine)

        # configure the training engine itself
        self._configure_train_sampler(engine=engine)
        self._configure_nan_callback(engine=engine)
        self._configure_cuda_cache_callback(engine=engine)
        self._configure_gpu_stats_callback(engine=engine)
        self._configure_time_profiler(engine=engine)
        self._configure_model_ema_callback(engine=engine)
        self._configure_metrics(engine=engine)
        self._configure_running_avg_logging(engine=engine)
        self._configure_progress_bar(engine=engine)
        self._configure_tb_logger(engine=engine)

        # configure the stuff where training engine and validation engine are connected
        self._configure_schedulers(engine=engine)
        self._configure_early_stopping_callback(engine=engine)
        self._configure_validation_engine(engine=engine)
        self._configure_visualization_engine(engine=engine)

        # configure model checkpointer
        self._configure_model_checkpointer(engine=engine)

        # print engine configuration info
        self._print_configuration_info()

        return engine

    def _print_configuration_info(self):
        """
        Prints the configuration information of the training engine.
        """
        logger.info("Configured training engine with the following parameters:")
        logger.info(f"\tOutput directory = {self._output_dir}")
        logger.info(f"\tDevice = {self._device}")
        logger.info(f"\tSync batch norm = {self._sync_batchnorm}")
        logger.info(f"\tBatch size = {self._dataloader.batch_size}")
        logger.info(f"\tTotal epochs = {self._max_epochs}")
        logger.info(f"\tEpoch length = {self._epoch_length}")
        logger.info(f"\tTotal steps per epoch = {self.batches_per_epoch}")
        logger.info(
            f"\tGradient accumulation per device = {self._gradient_config.gradient_accumulation_steps}"
        )
        logger.info(
            f"\tTotal optimizer update steps over epoch (scaled by grad accumulation steps) = {self.steps_per_epoch}"
        )
        logger.info(
            f"\tTotal optimizer update over complete training cycle (scaled by grad accumulation steps) = {self.total_update_steps}"
        )
        logger.info(f"\tTotal warmup steps = {self.total_warmup_steps}")
