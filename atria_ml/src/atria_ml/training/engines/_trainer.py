from __future__ import annotations

import copy
import inspect
import math
from collections import OrderedDict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from atria_logger import get_logger

from atria_ml.optimizers._base import OptimizerConfig
from atria_ml.optimizers._torch import SGDOptimizerConfig
from atria_ml.schedulers._base import LRSchedulerConfig
from atria_ml.task_pipelines._utilities import _find_checkpoint
from atria_ml.task_pipelines.configs._base import RunConfig
from atria_ml.training._configs import (
    GradientConfig,
    ModelCheckpointConfig,
    ModelEmaConfig,
    WarmupConfig,
)
from atria_ml.training.engine_steps._base import EngineStep
from atria_ml.training.engine_steps._training import TrainingStep
from atria_ml.training.engines._base import EngineBase, EngineConfig, EngineDependencies

if TYPE_CHECKING:
    import torch
    from ignite.engine import Engine, State

    from atria_ml.training.engines._validation_engine import ValidationEngine
    from atria_ml.training.handlers.ema_handler import EMAHandler

logger = get_logger(__name__)


class TrainerEngineDependencies(EngineDependencies):
    run_config: RunConfig


class TrainerEngineConfig(EngineConfig):
    clear_cuda_cache: bool | None = True
    stop_on_nan: bool = True
    eval_training: bool | None = False
    with_amp: bool = True
    max_epochs: int = 10
    validate_every_n_epochs: float = 1.0
    visualize_every_n_epochs: float = 1.0
    test_run: bool = False

    optimizer: dict[str, OptimizerConfig] | OptimizerConfig = SGDOptimizerConfig(
        lr=1e-3, momentum=0.9, weight_decay=0.0
    )
    lr_scheduler: dict[str, LRSchedulerConfig] | LRSchedulerConfig | None = None
    # lr_scheduler: dict[str, LRSchedulerConfig] | LRSchedulerConfig | None = (
    #     CosineAnnealingLRSchedulerConfig()
    # )
    model_ema: ModelEmaConfig = ModelEmaConfig()
    warmup: WarmupConfig = WarmupConfig()
    model_checkpoint: ModelCheckpointConfig = ModelCheckpointConfig()
    gradient: GradientConfig = GradientConfig()


class TrainerEngine(EngineBase[TrainerEngineConfig, TrainerEngineDependencies]):
    def __init__(self, config: TrainerEngineConfig, deps: TrainerEngineDependencies):
        self._validation_engine: ValidationEngine | None = None
        self._optimizers: dict[str, torch.optim.Optimizer] | None = None
        self._lr_schedulers: dict[str, torch.optim.lr_scheduler.LRScheduler] | None = (
            None
        )
        self._ema_handler: EMAHandler | None = None

        super().__init__(config, deps)

    @property
    def steps_per_epoch(self) -> int:
        """
        Returns the number of steps per epoch.

        Returns:
            int: Number of steps per epoch.
        """
        return (
            self.batches_per_epoch // self._config.gradient.gradient_accumulation_steps
        )

    @property
    def total_warmup_steps(self):
        """
        Returns the total number of warmup steps.

        Returns:
            int: Total number of warmup steps.
        """
        if self._config.warmup.warmup_steps is None:
            self._config.warmup.warmup_steps = 0
        if self._config.warmup.warmup_ratio is None:
            self._config.warmup.warmup_ratio = 0.0
        return (
            self._config.warmup.warmup_steps
            if self._config.warmup.warmup_steps > 0
            else math.ceil(self.total_update_steps * self._config.warmup.warmup_ratio)
        )

    @property
    def ema_handler(self) -> EMAHandler | None:
        return self._ema_handler

    def attach_validation_engine(self, validation_engine: ValidationEngine) -> None:
        self._validation_engine = validation_engine  # type: ignore

    def run(self, checkpoint_path: str | Path | None = None) -> State | None:
        if (
            checkpoint_path is None
            and self._config.model_checkpoint.resume_from_checkpoint
        ):
            # load resume checkpoint_path for training if none provided
            checkpoint_path = _find_checkpoint(
                output_dir=self._deps.output_dir, checkpoint_type="last"
            )

        # before running the engine log the first batch
        try:
            first_batch = next(iter(self._deps.dataloader))
            logger.info(
                f"First batch input for engine [{self.__class__.__name__}]: {first_batch}"
            )
        except Exception as e:
            logger.warning(
                f"Could not fetch the first batch from dataloader for engine [{self.__class__.__name__}]: {e}"
            )

        return super().run(checkpoint_path=checkpoint_path)

    def _build_engine(self) -> tuple[EngineStep, Engine]:
        # build optimizers
        self._optimizers = self._build_optimizers()

        # build lr schedulers
        self._lr_schedulers = self._build_lr_schedulers(self._optimizers)

        # log optimizers and lr schedulers
        for k, opt in self._optimizers.items():
            logger.info(f"Attached optimizer {k}={opt}")

        for k, sch in self._lr_schedulers.items():
            logger.info(f"Attached lr scheduler {k}={sch}")

        return super()._build_engine()

    def _build_optimizers(self) -> dict[str, torch.optim.Optimizer]:
        import ignite.distributed as idist

        trainable_parameters = copy.copy(self._deps.model_pipeline.trainable_parameters)
        optimizer_config_dict = (
            {"default": self._config.optimizer}
            if isinstance(self._config.optimizer, OptimizerConfig)
            else self._config.optimizer
        )
        assert len(trainable_parameters) == len(optimizer_config_dict), (
            "Number of optimizers must match the number of model parameter groups defined in the task_module. "
            f"Optimizers: {len(optimizer_config_dict)} != Model parameter groups: {len(trainable_parameters)}"
        )

        # build optimizers from configs
        optimizers = {}
        for k, opt_config in optimizer_config_dict.items():
            if k not in trainable_parameters.keys():
                raise ValueError(
                    f"Your optimizer configuration does not align with the model optimizer "
                    f"parameter groups. {k} =/= {trainable_parameters.keys()}"
                )

            # build base optimizer
            optimizer = opt_config.build(parameters=trainable_parameters[k])

            # initialize the optimizers from partial with the model parameters
            optimizer = idist.auto_optim(optimizer)

            # store optimizer
            optimizers[k] = optimizer
        return optimizers

    def _build_lr_schedulers(
        self, optimizers: dict[str, torch.optim.Optimizer]
    ) -> dict[str, torch.optim.lr_scheduler.LRScheduler]:
        if self._config.lr_scheduler is None:
            return {}

        lr_scheduler_config_dict = (
            {"default": self._config.lr_scheduler}
            if isinstance(self._config.lr_scheduler, LRSchedulerConfig)
            else self._config.lr_scheduler
        )
        lr_schedulers = {}
        for k, sch in lr_scheduler_config_dict.items():
            build_kwargs = {}
            sig = inspect.signature(sch.build)
            possible_args = set(sig.parameters.keys())
            if "total_update_steps" in possible_args:
                build_kwargs["total_update_steps"] = self.total_update_steps
            if "total_warmup_steps" in possible_args:
                build_kwargs["total_warmup_steps"] = self.total_warmup_steps
            if "steps_per_epoch" in possible_args:
                build_kwargs["steps_per_epoch"] = self.steps_per_epoch
            logger.debug(
                f"Initializing lr scheduler {sch} with runtime kwargs: {build_kwargs}"
            )
            lr_schedulers[k] = sch.build(optimizer=optimizers[k], **build_kwargs)
        return lr_schedulers

    def _build_engine_step(self) -> EngineStep:
        assert self._optimizers is not None, "Optimizers have not been built yet."
        return TrainingStep(
            model_pipeline=self._deps.model_pipeline,
            device=self._deps.device,
            optimizers=self._optimizers,
            gradient_config=self._config.gradient,
            with_amp=self._config.with_amp,
            test_run=self._config.test_run,
        )

    def _initialize_ignite_engine(self, engine_step: EngineStep) -> Engine:
        from ignite.engine import Engine

        class IgniteTrainingEngine(Engine):
            def state_dict(self) -> OrderedDict:
                state_dict = super().state_dict()
                if hasattr(self.state, "optimizer_step"):
                    state_dict["optimizer_step"] = self.state.optimizer_step  # type: ignore
                return state_dict

            def load_state_dict(self, state_dict: Mapping) -> None:
                super().load_state_dict(state_dict)
                if hasattr(self.state, "optimizer_step"):
                    self.state.optimizer_step = state_dict["optimizer_step"]  # type: ignore

        engine = IgniteTrainingEngine(engine_step)
        engine.logger.propagate = False
        return engine

    def _attach_handlers(self) -> None:
        self._register_events()

        self.attach_train_sampler()
        if self._config.stop_on_nan:
            self.attach_nan_callback()
        if self._config.clear_cuda_cache:
            self.attach_cuda_cache_callback()
        if self._config.model_ema.enabled:
            self.attach_model_ema_callback()

        # configure the stuff where training engine and validation engine are connected
        self.attach_schedulers()

        # configure model checkpointer
        self.attach_model_checkpointer()

        # attach parent
        engine = super()._attach_handlers()

        # print engine configuration info
        self._print_configuration_info()

        return engine

    def _attach_progress_bar(self) -> None:
        from ignite.engine import Events
        from ignite.handlers import ProgressBar

        # initialize the progress bar
        progress_bar = ProgressBar(
            desc=f"Stage [{self._engine_step.stage.value}]", persist=True
        )

        progress_bar.attach(
            self._engine,
            metric_names="all",
            event_name=Events.ITERATION_COMPLETED(
                every=self._config.logging.refresh_rate
            ),
            state_attributes=["optimizer_step", "ema_momentum"],
        )

        def _log_training_metrics(logger, epoch, elapsed, tag, metrics):
            metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
            logger.info(
                f"\nEpoch {epoch} - Training time (seconds): {elapsed:.2f} - {tag} metrics:\n {metrics_output}"
            )

        @self._engine.on(Events.EPOCH_COMPLETED)
        def progress_on_epoch_completed(engine: Engine) -> None:
            metrics = copy.deepcopy(engine.state.metrics)

            if hasattr(engine.state, "ema_momentum"):
                metrics["ema/mom"] = engine.state.ema_momentum  # type: ignore

            _log_training_metrics(
                logger=logger,
                epoch=engine.state.epoch,
                elapsed=engine.state.times["EPOCH_COMPLETED"],
                tag=self._engine_step.stage.value,
                metrics=metrics,
            )

        @self._engine.on(Events.TERMINATE | Events.INTERRUPT | Events.EXCEPTION_RAISED)
        def progress_on_terminate(engine: Engine) -> None:
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
                event_name=Events.ITERATION_COMPLETED(
                    every=self._config.logging.logging_steps
                ),
                tag="step",
                metric_names="all",
            )

            @self._engine.on(
                Events.TERMINATE | Events.INTERRUPT | Events.EXCEPTION_RAISED
            )
            def on_terminate(engine: Engine) -> None:
                if self._deps.tb_logger is not None:
                    self._deps.tb_logger.close()

            # attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at every
            # 'logging_steps' iteration
            assert self._optimizers is not None, "Optimizers have not been built yet."
            for param_name in ["lr", "weight_decay"]:
                for k, opt in self._optimizers.items():
                    self._deps.tb_logger.attach_opt_params_handler(
                        self._engine,
                        event_name=Events.ITERATION_STARTED(
                            every=self._config.logging.logging_steps
                        ),
                        optimizer=opt,
                        param_name=param_name,
                        tag=f"step/opt/{k}",
                    )

    def attach_train_sampler(self):
        import ignite.distributed as idist
        from torch.utils.data.distributed import DistributedSampler

        if idist.get_world_size() > 1:
            from ignite.engine import Events

            train_sampler = self._deps.dataloader.sampler
            if not isinstance(train_sampler, DistributedSampler):
                raise TypeError(
                    "Train sampler should be torch DistributedSampler and have `set_epoch` method"
                )

            @self._engine.on(Events.EPOCH_STARTED)
            def distrib_set_epoch(engine: Engine) -> None:
                cast(DistributedSampler, train_sampler).set_epoch(
                    engine.state.epoch - 1
                )

        else:
            # check whether the correct training sample is being used
            if self._deps.dataloader.sampler is not None and isinstance(
                self._deps.dataloader.sampler, DistributedSampler
            ):
                logger.warning(
                    "Argument train_sampler is a distributed sampler,"
                    " but either there is no distributed setting or world size is < 2. "
                    "Train sampler argument will be ignored",
                    UserWarning,
                )

    def attach_nan_callback(self):
        from ignite.engine import Events

        from atria_ml.training.handlers.terminate_on_nan import TerminateOnNan

        self._engine.add_event_handler(
            Events.ITERATION_COMPLETED,
            TerminateOnNan(output_transform=lambda x: x.__dict__),
        )

    def attach_cuda_cache_callback(self):
        from ignite.engine import Events

        from atria_ml.training.handlers.terminate_on_nan import TerminateOnNan

        self._engine.add_event_handler(
            Events.ITERATION_COMPLETED,
            TerminateOnNan(output_transform=lambda x: x.__dict__),
        )

    def attach_model_ema_callback(self) -> None:
        from atria_models.utilities._ddp_model_proxy import ModuleProxyWrapper
        from torchinfo import summary

        from atria_ml.training.engines._events import OptimizerEvents
        from atria_ml.training.handlers.ema_handler import EMAHandler

        trainable_model = self._deps.model_pipeline._model
        if isinstance(trainable_model, ModuleProxyWrapper):
            trainable_model = trainable_model.module

        self._ema_handler = EMAHandler(
            trainable_model,
            momentum=self._config.model_ema.momentum,
            momentum_warmup=self._config.model_ema.momentum_warmup,
            warmup_iters=self._config.model_ema.warmup_iters,
            handle_buffers="update",
        )

        logger.info(
            f"Attaching EMAHandler with following configuration: {self._config.model_ema}"
        )
        logger.info("Ema Model:")
        logger.info(summary(self._ema_handler.ema_model, verbose=0, depth=2))
        self._ema_handler.attach(
            self._engine,
            name="ema_momentum",
            event=OptimizerEvents.optimizer_step(
                every=self._config.model_ema.update_every
            ),
        )

    def attach_schedulers(self) -> None:
        from ignite.engine import Events
        from ignite.handlers import (
            LRScheduler,
            ParamScheduler,
            ReduceLROnPlateauScheduler,
            create_lr_scheduler_with_warmup,
        )
        from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, StepLR

        from atria_ml.training.engines._events import OptimizerEvents

        if self._lr_schedulers is None or len(self._lr_schedulers) == 0:
            return

        for k, inner_sch in self._lr_schedulers.items():
            if self.total_warmup_steps > 0:
                logger.info(
                    f"Initialized lr scheduler {inner_sch.__class__.__name__} with warmup. "
                )
                logger.info(f"Warmup ratio = {self._config.warmup.warmup_ratio}. ")
                logger.info(
                    f"Number of warmup steps = {self.total_warmup_steps}. This corresponds to optimizer updates, "
                    "not total batches in epoch and therefore its scaled by grad "
                    f"acummulation steps = {self._config.gradient.gradient_accumulation_steps}."
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
                        event_filter=lambda _, __: self._engine.state.optimizer_step  # type: ignore
                        <= self.total_warmup_steps
                    )

                    # Trigger scheduler on epoch_started events after the warm-up. Epochs are 1-based, thus we do 1 +
                    combined_events |= Events.EPOCH_STARTED(
                        event_filter=lambda _, __: self._engine.state.epoch
                        > 1 + self.total_warmup_steps / self.steps_per_epoch
                    )

                    self._engine.add_event_handler(combined_events, sch)

                    # update scheduler in dict
                    self._lr_schedulers[k] = sch  # type: ignore
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
                    self._engine.add_event_handler(
                        OptimizerEvents.optimizer_step(
                            event_filter=lambda _, __: self._engine.state.optimizer_step  # type: ignore
                            <= self.total_warmup_steps
                        ),
                        sch.schedulers[0],
                    )

                    # Trigger scheduler on epoch_started events after the warm-up. Epochs are 1-based, thus we do 1 +
                    combined_events = Events.COMPLETED | Events.COMPLETED(
                        event_filter=lambda _, __: self._engine.state.epoch
                        > 1 + self.total_warmup_steps / self.steps_per_epoch
                    )

                    assert self._validation_engine is not None, (
                        "Validation engine must be attached for ReduceLROnPlateauScheduler."
                    )
                    self._validation_engine._engine.add_event_handler(
                        combined_events, inner_sch
                    )
                else:
                    logger.info(
                        "Both warmup updates and the scheduler updates are triggered per optimizer step."
                    )
                    sch = create_lr_scheduler_with_warmup(
                        inner_sch,
                        warmup_start_value=0.0,
                        warmup_duration=self.total_warmup_steps,
                    )
                    self._engine.add_event_handler(OptimizerEvents.optimizer_step, sch)
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
                    self._engine.add_event_handler(Events.EPOCH_STARTED, sch)
                elif isinstance(inner_sch, ReduceLROnPlateauScheduler):
                    logger.info(
                        f"Initialized lr scheduler {inner_sch.__class__.__name__}. Scheduler updates are triggered per validation step. "
                    )
                    # inner_sch.trainer = training_engine
                    self._engine.add_event_handler(Events.COMPLETED, sch)
                else:
                    logger.info(
                        f"Initialized lr scheduler {inner_sch.__class__.__name__}. Scheduler updates are triggered per optimizer step. "
                    )
                    self._engine.add_event_handler(OptimizerEvents.optimizer_step, sch)

    def _to_load_state_dict(self) -> dict[str, Any]:
        from atria_ml.training.engines.utilities import (
            MODEL_PIPELINE_CHECKPOINT_KEY,
            TRAINING_ENGINE_KEY,
        )

        checkpoint_state_dict = {
            MODEL_PIPELINE_CHECKPOINT_KEY: self._deps.model_pipeline,
            TRAINING_ENGINE_KEY: self._engine,
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

        return checkpoint_state_dict

    def _to_save_state_dict(self) -> dict[str, Any]:
        from atria_ml.training.engines.utilities import (
            CONFIG_KEY,
            MODEL_PIPELINE_CHECKPOINT_KEY,
            TRAINING_ENGINE_KEY,
        )

        checkpoint_state_dict = {
            CONFIG_KEY: self._deps.run_config,
            MODEL_PIPELINE_CHECKPOINT_KEY: self._deps.model_pipeline,
            TRAINING_ENGINE_KEY: self._engine,
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

        return checkpoint_state_dict

    def attach_model_checkpointer(self):
        from ignite.engine import Events
        from ignite.handlers import DiskSaver
        from ignite.handlers.checkpoint import BaseSaveHandler, Checkpoint

        # setup checkpoint saving if required
        if self._config.model_checkpoint:
            logger.info("Configuring model checkpointing with the following config:")
            logger.info(f"{self._config.model_checkpoint}")
            checkpoint_state_dict = self._to_save_state_dict()

            checkpoint_dir = (
                Path(self._deps.output_dir) / self._config.model_checkpoint.dir
            )
            save_handler = DiskSaver(checkpoint_dir, require_empty=False)
            if self._config.model_checkpoint.save_per_epoch:
                checkpoint_handler = Checkpoint(
                    checkpoint_state_dict,
                    cast(Callable | BaseSaveHandler, save_handler),
                    filename_prefix=self._config.model_checkpoint.name_prefix,
                    global_step_transform=lambda *_: self._engine.state.epoch,
                    n_saved=self._config.model_checkpoint.n_saved,
                    # include_self=True,
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
                    # include_self=True,
                )
                self._engine.add_event_handler(
                    Events.ITERATION_COMPLETED(
                        every=self._config.model_checkpoint.save_every_iters
                    )
                    | Events.COMPLETED,
                    checkpoint_handler,
                )

    def _register_events(self) -> None:
        from atria_ml.training.engines._events import OptimizerEvents

        self._engine.register_events(
            *OptimizerEvents,
            event_to_attr={
                OptimizerEvents.optimizer_step: OptimizerEvents.optimizer_step.value
            },
        )

    def _print_configuration_info(self):
        """
        Prints the configuration information of the training engine.
        """
        logger.info("Configured training engine with the following parameters:")
        logger.info(f"\tOutput directory = {self._deps.output_dir}")
        logger.info(f"\tDevice = {self._deps.device}")
        logger.info(f"\tBatch size = {self._deps.dataloader.batch_size}")
        logger.info(f"\tTotal epochs = {self._config.max_epochs}")
        logger.info(f"\tEpoch length = {self._config.epoch_length}")
        logger.info(f"\tTotal steps per epoch = {self.batches_per_epoch}")
        logger.info(
            f"\tGradient accumulation per device = {self._config.gradient.gradient_accumulation_steps}"
        )
        logger.info(
            f"\tTotal optimizer update steps over epoch (scaled by grad accumulation steps) = {self.steps_per_epoch}"
        )
        logger.info(
            f"\tTotal optimizer update over complete training cycle (scaled by grad accumulation steps) = {self.total_update_steps}"
        )
        logger.info(f"\tTotal warmup steps = {self.total_warmup_steps}")
