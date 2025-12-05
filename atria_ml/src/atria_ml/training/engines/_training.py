from __future__ import annotations

import copy
import math
from collections import OrderedDict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from atria_logger import get_logger

from atria_ml.training.configs.early_stopping_config import EarlyStoppingConfig
from atria_ml.training.configs.gradient_config import GradientConfig
from atria_ml.training.configs.model_checkpoint import ModelCheckpointConfig
from atria_ml.training.configs.model_ema_config import ModelEmaConfig
from atria_ml.training.configs.warmup_config import WarmupConfig
from atria_ml.training.engine_builders._base import EngineBase
from atria_ml.training.engine_steps._base import EngineStep
from atria_ml.training.engine_steps._training import TrainingStep

if TYPE_CHECKING:
    import torch
    from ignite.engine import Engine, State
    from ignite.handlers import TensorboardLogger

logger = get_logger(__name__)


class TrainingEngine(EngineBase):
    def __init__(
        self,
        *args,
        run_config: dict[str, Any],
        optimizers: dict[str, torch.optim.Optimizer],
        lr_schedulers: dict[str, torch.optim.lr_scheduler.LRScheduler] | None = None,
        eval_training: bool | None = False,
        stop_on_nan: bool = True,
        clear_cuda_cache: bool | None = True,
        model_ema_config: ModelEmaConfig = ModelEmaConfig(),
        warmup_config: WarmupConfig = WarmupConfig(),
        early_stopping: EarlyStoppingConfig = EarlyStoppingConfig(),
        model_checkpoint_config: ModelCheckpointConfig = ModelCheckpointConfig(),
        gradient_config: GradientConfig = GradientConfig(),
        with_amp: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._run_config = run_config
        self._optimizers = optimizers
        self._lr_schedulers = lr_schedulers or {}
        self._eval_training = eval_training
        self._stop_on_nan = stop_on_nan
        self._clear_cuda_cache = clear_cuda_cache
        self._model_ema_config = model_ema_config
        self._warmup_config = warmup_config
        self._early_stopping = early_stopping
        self._model_checkpoint_config = model_checkpoint_config
        self._gradient_config = gradient_config
        self._with_amp = with_amp

    def _build_engine_step(self) -> EngineStep:
        return TrainingStep(
            model_pipeline=self._model_pipeline,
            device=self._device,
            optimizers=self._optimizers,
            gradient_config=self._gradient_config,
            with_amp=self._with_amp,
            test_run=self._test_run,
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

    def build(self) -> Engine:
        # optimized_parameters_dict = self._model_pipeline.trainable_parameters
        # assert len(self._model_pipeline.trainable_parameters) == len(
        #     self._optimizers
        # ), (
        #     "Number of optimizers must match the number of model parameter groups defined in the task_module. "
        #     f"Optimizers: {len(self._optimizers)} != Model parameter groups: {len(optimized_parameters_dict)}"
        # )

        # self._optimizers = {}
        # for k, opt in self._optimizers.items():
        #     if k not in optimized_parameters_dict.keys():
        #         raise ValueError(
        #             f"Your optimizer configuration does not align with the model optimizer "
        #             f"parameter groups. {k} =/= {optimized_parameters_dict.keys()}"
        #         )

        #     # initialize the optimizers from partial with the model parameters
        #     self._optimizers[k] = idist.auto_optim(
        #         opt(params=optimized_parameters_dict[k])
        #     )

        # print information
        # _print_optimizers_info(self._optimizers)

        # # initialize lr schedulers partials
        # self._lr_schedulers = {}
        # if self._lr_scheduler is not None:
        #     for k, sch in self._lr_scheduler.items():
        #         runtime_kwargs = {}
        #         for kwarg in [
        #             "total_update_steps",
        #             "total_warmup_steps",
        #             "steps_per_epoch",
        #         ]:
        #             if kwarg in sch.get_possible_args():
        #                 runtime_kwargs[kwarg] = getattr(self, kwarg)

        #         logger.debug(
        #             f"Initializing lr scheduler {sch} with runtime kwargs: {runtime_kwargs}"
        #         )
        #         self._lr_schedulers[k] = sch(
        #             optimizer=self._optimizers[k], **runtime_kwargs
        #         )

        #     # print information
        #     _print_schedulers_info(self._lr_schedulers)

        return super().build()

    def run(self, checkpoint_path: str | Path | None = None) -> State | None:
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
            return None

        return self._engine.run(
            (
                FixedBatchIterator(self._dataloader, self._dataloader.batch_size)
                if self._use_fixed_batch_iterator
                else self._dataloader
            ),
            max_epochs=self._max_epochs,
            epoch_length=self._epoch_length,
        )

    def _initialize_ignite_engine(self, engine_step: EngineStep) -> Engine:
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

        engine = IgniteTrainingEngine(engine_step)
        engine.logger.propagate = False
        return engine

    def attach_handlers(self, engine: Engine, stage: str) -> Engine:
        engine = super().attach_handlers(engine=engine, stage=stage)

        self._register_events(engine=engine)

        self.attach_train_sampler(engine=engine)
        if self._stop_on_nan:
            self.attach_nan_callback(engine=engine)
        if self._clear_cuda_cache:
            self.attach_cuda_cache_callback(engine=engine)
        if self._model_ema_config.enabled:
            self.attach_model_ema_callback(engine=engine)  # type: ignore

        # configure the stuff where training engine and validation engine are connected
        self.attach_schedulers(engine=engine)
        self.attach_early_stopping_callback(engine=engine)

        # configure model checkpointer
        self.attach_model_checkpointer(engine=engine)

        # print engine configuration info
        self._print_configuration_info()

        return engine

    def attach_progress_bar(self, engine: Engine, stage: str) -> None:
        from ignite.engine import Events
        from ignite.handlers import ProgressBar

        from atria_ml.training.engines.utilities import _log_training_metrics

        # initialize the progress bar
        progress_bar = ProgressBar(desc=f"Stage [{stage}]", persist=True)

        progress_bar.attach(
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
                tag=stage,
                metrics=metrics,
            )

        @engine.on(Events.TERMINATE | Events.INTERRUPT | Events.EXCEPTION_RAISED)
        def progress_on_terminate(engine: Engine) -> None:
            progress_bar.close()

    def attach_tb_logger(self, engine: Engine, tb_logger: TensorboardLogger):
        import ignite.distributed as idist
        from ignite.engine import Events

        if idist.get_rank() == 0 and tb_logger is not None and self._logging.log_to_tb:
            tb_logger.attach_output_handler(
                engine,
                event_name=Events.ITERATION_COMPLETED(
                    every=self._logging.logging_steps
                ),
                tag="step",
                metric_names="all",
            )

            @engine.on(Events.TERMINATE | Events.INTERRUPT | Events.EXCEPTION_RAISED)
            def on_terminate(engine: Engine) -> None:
                tb_logger.close()

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

    def attach_train_sampler(self, engine: Engine):
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

    def attach_nan_callback(self, engine: Engine):
        from ignite.engine import Events

        from atria_ml.training.handlers.terminate_on_nan import TerminateOnNan

        engine.add_event_handler(
            Events.ITERATION_COMPLETED,
            TerminateOnNan(output_transform=lambda x: x.__dict__),
        )

    def attach_cuda_cache_callback(self, engine: Engine):
        from ignite.engine import Events

        from atria_ml.training.handlers.terminate_on_nan import TerminateOnNan

        engine.add_event_handler(
            Events.ITERATION_COMPLETED,
            TerminateOnNan(output_transform=lambda x: x.__dict__),
        )

    def attach_model_ema_callback(self, engine: Engine) -> None:
        from atria_models.utilities._ddp_model_proxy import ModuleProxyWrapper
        from torchinfo import summary

        from atria_ml.training.engines.events import OptimizerEvents
        from atria_ml.training.handlers.ema_handler import EMAHandler

        trainable_model = self._model_pipeline._model
        if isinstance(trainable_model, ModuleProxyWrapper):
            trainable_model = trainable_model.module

        self._ema_handler = EMAHandler(
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

    def attach_schedulers(self, engine: Engine) -> None:
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

    def _to_load_checkpoint(self) -> dict[str, Any]:
        from atria_ml.training.engines.utilities import (
            MODEL_PIPELINE_CHECKPOINT_KEY,
            RUN_CONFIG_KEY,
            TRAINING_ENGINE_KEY,
        )

        checkpoint_state_dict = {
            RUN_CONFIG_KEY: self._run_config,
            TRAINING_ENGINE_KEY: self._engine,
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

        return checkpoint_state_dict

    def attach_model_checkpointer(self, engine: Engine):
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

    def attach_early_stopping_callback(self, engine: Engine) -> None:
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
        from atria_ml.training.engines.events import OptimizerEvents

        engine.register_events(
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
