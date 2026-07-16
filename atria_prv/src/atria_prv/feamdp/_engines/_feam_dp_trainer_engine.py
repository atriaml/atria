from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from atria_logger import get_logger
from atria_ml.optimizers._configs import SGDOptimizerConfig
from atria_ml.training.engine_steps._base import EngineStep
from atria_ml.training.engine_steps._training import TrainingStep
from atria_ml.training.engines._base import EngineBase, EngineConfig, EngineDependencies
from opacus.utils.batch_memory_manager import BatchMemoryManager

from atria_prv.dp.configs import DPConfig
from atria_prv.dp.opacus.privacy_engine import _PrivacyEngine

if TYPE_CHECKING:
    from ignite.engine import Engine, State

logger = get_logger(__name__)


class FeAmDPTrainerEngineDependencies(EngineDependencies):
    pass


class FeAmDPTrainerEngineConfig(EngineConfig):
    dp_config: DPConfig


class FeAmDPTrainerEngine(
    EngineBase[FeAmDPTrainerEngineConfig, FeAmDPTrainerEngineDependencies]
):
    def __init__(
        self, config: FeAmDPTrainerEngineConfig, deps: FeAmDPTrainerEngineDependencies
    ):
        super().__init__(config=config, deps=deps)
        self._config: FeAmDPTrainerEngineConfig
        self._deps: FeAmDPTrainerEngineDependencies

    @property
    def batches_per_epoch(self) -> int:
        if self._config.dp_config.use_bmm:
            ratio = max(
                self._deps.dataloader.batch_size
                / self._config.dp_config.max_physical_batch_size,
                1.0,
            )
            return len(self._deps.dataloader) * ratio
        else:
            return len(self._deps.dataloader)

    def _register_events(self) -> None:
        from atria_ml.training.engines._events import OptimizerEvents

        self._engine.register_events(
            *OptimizerEvents,  # type: ignore[arg-type]
            event_to_attr={
                OptimizerEvents.optimizer_step: OptimizerEvents.optimizer_step.value
            },
        )

    def _build_optimizers(self) -> dict[str, torch.optim.Optimizer]:
        import ignite.distributed as idist

        trainable_parameters = self._deps.model_pipeline.trainable_parameters
        optimizer_config_dict = {"default": SGDOptimizerConfig(lr=1.0, momentum=0.0)}
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

    def _build_engine(self) -> tuple[EngineStep, Engine]:
        # build optimizers
        self._optimizers = self._build_optimizers()

        # log optimizers and lr schedulers
        for k, opt in self._optimizers.items():
            logger.info(f"Attached optimizer {k}={opt}")

        # assert target delta is set for clients
        assert self._config.dp_config.target_delta is not None, (
            "target_delta must be passed to the client trainer."
        )

        # privacy trainer only supports a single optimizer for now
        optimizer_keys = list(self._optimizers.keys())
        assert len(optimizer_keys) == 1 and "default" in optimizer_keys

        # put model in training mode
        self._deps.model_pipeline._model.train()

        self._privacy_engine = _PrivacyEngine(
            accountant=self._config.dp_config.accountant
        )
        logger.info(f"Setting client dp engine with {self._config.dp_config}")
        self._hooks, self._optimizers["default"], self._dp_dataloader = (
            self._privacy_engine.make_private(
                module=self._deps.model_pipeline._model,
                optimizer=self._optimizers["default"],
                data_loader=self._deps.dataloader,
                noise_multiplier=self._config.dp_config.noise_multiplier,
                max_grad_norm=self._config.dp_config.max_grad_norm,
                sample_rate=self._config.dp_config.sample_rate,
                wrap_model=False,
            )
        )
        return EngineBase._build_engine(self)

    def _build_engine_step(self) -> EngineStep:
        assert self._optimizers is not None, "Optimizers have not been built yet."
        return TrainingStep(
            model_pipeline=self._deps.model_pipeline,
            device=self._deps.device,
            optimizers=self._optimizers,
            test_run=self._config.test_run,
        )

    def _attach_handlers(self) -> None:
        from ignite.engine import Events

        # configure engine
        self._register_events()
        self._setup_test_run()
        self._attach_progress_bar()

        self._metrics = None
        self._attach_metrics()

        def terminate_on_optimizer_step(
            engine,
        ):  # this is necessary for fldp to work with correct privacy accounting
            if not self._optimizers["default"]._is_last_step_skipped:
                logger.info("_is_last_step_skipped received. Terminating...")
                engine.terminate()

        self._engine.add_event_handler(
            Events.ITERATION_COMPLETED, terminate_on_optimizer_step
        )

        @self._engine.on(Events.TERMINATE | Events.INTERRUPT)
        def on_terminate(engine: Engine) -> None:
            logger.info(
                f"Engine [{self.__class__.__name__}] terminated after {engine.state.epoch} epochs."
            )

        # add handler for exception
        @self._engine.on(Events.EXCEPTION_RAISED)
        def on_exception(exception: Exception) -> None:
            raise exception

    def _print_configuration_info(self):
        """
        Prints the configuration information of the training engine.
        """
        logger.info("Configured training engine with the following parameters:")
        logger.info(f"\tOutput directory = {self._deps.output_dir}")
        logger.info(f"\tDevice = {self._deps.device}")
        logger.info(f"\tBatch size = {self._deps.dataloader.batch_size}")
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
        for k, opt in self._optimizers.items():
            logger.info(
                f"Using sigma[{k}]={opt.noise_multiplier} and C={self._config.dp_config.max_grad_norm}"
            )

    def run(self) -> State | None:
        # run engine
        if self._deps.output_dir is not None:
            logger.info(
                f"Running {self.__class__.__name__} engine with batch size [{self._deps.dataloader.batch_size}] and output_dir: {self._deps.output_dir}"
            )
        else:
            logger.info(f"Running engine {self.__class__.__name__}.")

        # move model pipeline to device
        self._deps.model_pipeline.ops.to_device(self._deps.device)

        if self._config.dp_config.use_bmm:
            with BatchMemoryManager(
                data_loader=self._dp_dataloader,
                max_physical_batch_size=self._config.dp_config.max_physical_batch_size,
                optimizer=self._optimizers["default"],
            ) as memory_safe_data_loader:
                logger.info("Running trainer on memory safe dataloader.")
                state = self._engine.run(memory_safe_data_loader, max_epochs=1)
        else:
            state = self._engine.run(self._dp_dataloader, max_epochs=1)

        self._hooks.cleanup()
        return state
