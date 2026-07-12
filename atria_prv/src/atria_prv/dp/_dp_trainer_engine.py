from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from atria_logger import get_logger
from atria_ml.task_pipelines._utilities import _find_checkpoint
from atria_ml.training.engine_steps._base import EngineStep
from atria_ml.training.engines._base import EngineBase
from atria_ml.training.engines._trainer import (
    TrainerEngine,
    TrainerEngineConfig,
    TrainerEngineDependencies,
)
from opacus.utils.batch_memory_manager import BatchMemoryManager

from atria_prv.dp.configs import DPConfig
from atria_prv.dp.opacus.metrics import PrivacyLossMetric
from atria_prv.dp.opacus.privacy_engine import _PrivacyEngine

if TYPE_CHECKING:
    from ignite.engine import Engine, State

logger = get_logger(__name__)


class DPTrainerEngineDependencies(TrainerEngineDependencies):
    privacy_engine: _PrivacyEngine


class DPTrainerEngineConfig(TrainerEngineConfig):
    dp_config: DPConfig


class DPTrainerEngine(TrainerEngine):
    def __init__(
        self, config: DPTrainerEngineConfig, deps: DPTrainerEngineDependencies
    ):
        super().__init__(config=config, deps=deps)
        self._config: DPTrainerEngineConfig
        self._deps: DPTrainerEngineDependencies

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

        # set delta inverse of dataset size
        if self._config.dp_config.target_delta is None:
            total_dataset_samples = len(self._deps.dataloader.dataset)
            self._target_delta = 1.0 / total_dataset_samples
        else:
            self._target_delta = self._config.dp_config
        logger.info(
            f"Setting privacy delta for total samples [{total_dataset_samples}] = {self._target_delta}"
        )

        # privacy trainer only supports a single optimizer for now
        optimizer_keys = list(self._optimizers.keys())
        assert len(optimizer_keys) == 1 and "default" in optimizer_keys

        # put model in training mode
        self._deps.model_pipeline._model.train()

        # setup privacy
        if self._config.dp_config.target_epsilon is not None:
            (
                self._hooks,
                self._optimizers[
                    "default"
                ],  # dp only supports one optimizer at this time
                self._dp_dataloader,
            ) = self._deps.privacy_engine.make_private_with_epsilon(
                module=self._deps.model_pipeline._model,
                optimizer=self._optimizers[
                    "default"
                ],  # dp only supports one optimizer at this time
                data_loader=self._deps.dataloader,
                epochs=self._config.max_epochs,
                target_epsilon=self._config.dp_config.target_epsilon,
                target_delta=self._target_delta,
                max_grad_norm=self._config.dp_config.max_grad_norm,
                wrap_model=False,
                # clipping="per_layer",
            )
        else:
            (
                self._hooks,
                self._optimizers[
                    "default"
                ],  # dp only supports one optimizer at this time
                self._dp_dataloader,
            ) = self._deps.privacy_engine.make_private(
                module=self._deps.model_pipeline._model,
                optimizer=self._optimizers[
                    "default"
                ],  # dp only supports one optimizer at this time
                data_loader=self._deps.dataloader,
                noise_multiplier=self._config.dp_config.noise_multiplier,
                max_grad_norm=self._config.dp_config.max_grad_norm,
                wrap_model=False,
            )

        for k, opt in self._optimizers.items():
            logger.info(
                f"Using sigma[{k}]={opt.noise_multiplier} and C={self._config.dp_config.max_grad_norm}"
            )

        return EngineBase._build_engine(self)

    def _attach_handlers(self) -> None:
        from ignite.metrics import BatchWise

        metric = PrivacyLossMetric(
            privacy_engine=self._deps.privacy_engine, delta=self._target_delta
        )
        metric.attach(self._engine, name="p_loss", usage=BatchWise())

        super()._attach_handlers()

    def _to_load_state_dict(self) -> dict[str, Any]:
        checkpoint_state_dict = super()._to_load_state_dict()
        checkpoint_state_dict["privacy_engine"] = self._deps.privacy_engine
        return checkpoint_state_dict

    def _to_save_state_dict(self) -> dict[str, Any]:
        checkpoint_state_dict = super()._to_save_state_dict()
        checkpoint_state_dict["privacy_engine"] = self._deps.privacy_engine
        return checkpoint_state_dict

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

    def run(self, checkpoint_path: str | Path | None = None) -> State | None:
        # run engine
        if self._deps.output_dir is not None:
            logger.info(
                f"Running {self.__class__.__name__} engine with batch size [{self._deps.dataloader.batch_size}] and output_dir: {self._deps.output_dir}"
            )
        else:
            logger.info(f"Running engine {self.__class__.__name__}.")

        # move model pipeline to device
        self._deps.model_pipeline.ops.to_device(self._deps.device)

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
            logger.info(f"First batch input for engine [{self.__class__.__name__}]:")
            total_elements = len(first_batch)
            first_sample = first_batch[0]
            logger.info(
                f"\tTotal elements in the batch: {total_elements}, First sample input: {first_sample}"
            )
        except Exception as e:
            logger.warning(
                f"Could not fetch the first batch from dataloader for engine [{self.__class__.__name__}]: {e}"
            )

        # load checkpoint if provided
        if checkpoint_path is not None:
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            self._load_checkpoint(checkpoint_path=checkpoint_path)

            resume_epoch = self._engine.state.epoch
            if (
                self._engine._is_done(self._engine.state)
                and resume_epoch >= self._config.max_epochs
            ):  # if we are resuming from last checkpoint and training is already finished
                logger.warning(
                    f"{self.__class__.__name__} has already been finished! Either increase the number of "
                    f"epochs (current={self._config.max_epochs}) >= {resume_epoch} "
                    "OR reset the training from start."
                )
                return

            logger.info(
                f"Resuming {self.__class__.__name__} engine with checkpoint: {checkpoint_path}."
            )

        if self._config.dp_config.use_bmm:
            with BatchMemoryManager(
                data_loader=self._dp_dataloader,
                max_physical_batch_size=self._config.dp_config.max_physical_batch_size,
                optimizer=self._optimizers["default"],
            ) as memory_safe_data_loader:
                state = self._engine.run(
                    memory_safe_data_loader,
                    max_epochs=self._config.max_epochs,
                    epoch_length=self._config.epoch_length,
                )
        else:
            state = self._engine.run(
                self._dp_dataloader,
                max_epochs=self._config.max_epochs,
                epoch_length=self._config.epoch_length,
            )

        self._hooks.cleanup()
        return state
