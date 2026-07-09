from __future__ import annotations

import logging
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa
from atria_logger import get_logger
from atria_logger._api import get_logger
from atria_ml.data_pipeline._data_pipeline import DataPipeline
from atria_ml.task_pipelines._trainer import Trainer, TrainerState
from atria_ml.task_pipelines._utilities import _initialize_torch
from atria_ml.training.engines._trainer import (
    TrainerEngine,
    TrainerEngineConfig,
    TrainerEngineDependencies,
)

if TYPE_CHECKING:
    from ignite.handlers import TensorboardLogger

    from atria_prv.configs import FLTrainingTaskConfig

logger = get_logger(__name__)


@contextmanager
def suppress_logging(level=logging.CRITICAL):
    """Suppress all logging at or below the given level."""
    logging.disable(level)
    try:
        yield
    finally:
        logging.disable(logging.NOTSET)  # Re-enable all logging


@dataclass
class FLClientOutput:
    params: OrderedDict[str, torch.Tensor]
    metrics: dict | None = None


@dataclass
class FLClientTrainerState(TrainerState):
    client_id: int | None = None


class FLClientTrainer(Trainer):
    def __init__(
        self,
        config: FLTrainingTaskConfig,
        client_id: int,
        partition_cache_dir: str | None,
    ) -> None:
        self._config = config
        self._client_id = client_id
        self._partition_cache_dir = partition_cache_dir
        self._state: FLClientTrainerState = self._build()

    def _initialize_runtime(self) -> None:
        import ignite.distributed as idist
        import torch

        # initialize training
        _initialize_torch(
            seed=self._config.env.seed + self._client_id + 1,
            deterministic=self._config.env.deterministic,
        )

        # initialize torch device (cpu or gpu)
        if torch.cuda.is_available():
            self._device = idist.device()
        else:
            self._device = "cpu"

        logger.info(
            f"Seed set to {self._config.env.seed + self._client_id + 1} and device to {self._device} on client {self._client_id}."
        )

    def _setup_logging(self) -> TensorboardLogger | None:
        pass

    def _build(self) -> TrainerState:
        self._initialize_runtime()

        # setup logging
        tb_logger = self._setup_logging()

        # get model transforms
        train_transform = self._config.model_pipeline.train_transform
        eval_transform = self._config.model_pipeline.eval_transform

        # build dataset
        dataset = self._config.data.build_dataset()

        # apply transforms
        dataset.apply_transforms(
            train_transform=train_transform, eval_transform=eval_transform
        )

        # load labels
        labels = dataset.metadata.dataset_labels

        # log dataset info
        logger.info(
            f"Dataset [Train]: {len(dataset.train)}, [Validation]: {len(dataset.validation)}, [Test]: {len(dataset.test)}"
        )

        # build model pipeline
        with suppress_logging():
            model_pipeline = self._config.model_pipeline.build(labels=labels)

            # build data pipeline
            data_pipeline = DataPipeline(dataset=dataset)

        return FLClientTrainerState(
            data_pipeline=data_pipeline,
            model_pipeline=model_pipeline,
            tb_logger=tb_logger,
            client_id=self._client_id,
        )

    def _build_train_engine(self) -> TrainerEngine:
        """Overridable seam: a future DPFLClient overrides ONLY this method (swapping in
        DPTrainerEngine + a privacy engine), mirroring how DPTrainer overrides
        Trainer._build_train_engine today. Nothing else on FLClient needs to change.

        Naming note: this runs `trainer.max_epochs` local epochs by default, but
        `client_update`'s contract doesn't assume that — a different local-update
        strategy (e.g. a single local step) only needs to change what happens inside
        this method, not the outer round loop.
        """
        train_dataloader = self._state.data_pipeline.train_dataloader(
            batch_size=self._config.data.train_batch_size,
            num_workers=self._config.data.num_workers,
            pin_memory=self._config.data.pin_memory,
        )
        # A client round is ephemeral: only the resulting parameters are returned to the
        # server for aggregation, so disable checkpointing entirely (nothing to persist,
        # output_dir=None) and turn off resume-from-checkpoint discovery.
        no_checkpoint_cfg = self._config.trainer.model_checkpoint.model_copy(
            update={"enabled": False, "resume_from_checkpoint": False}
        )
        return TrainerEngine(
            config=TrainerEngineConfig(
                max_epochs=self._config.trainer.max_epochs,
                outputs_to_running_avg=self._config.trainer.outputs_to_running_avg,
                logging=self._config.logging,
                test_run=self._config.test_run,
                use_fixed_batch_iterator=self._config.use_fixed_batch_iterator,
                with_amp=self._config.with_amp,
                clear_cuda_cache=self._config.trainer.clear_cuda_cache,
                stop_on_nan=self._config.trainer.stop_on_nan,
                eval_training=self._config.trainer.eval_training,
                validate_every_n_epochs=self._config.trainer.validate_every_n_epochs,
                visualize_every_n_epochs=self._config.trainer.visualize_every_n_epochs,
                optimizer=self._config.trainer.optimizer,
                lr_scheduler=self._config.trainer.lr_scheduler,
                model_ema=self._config.trainer.model_ema,
                warmup=self._config.trainer.warmup,
                model_checkpoint=no_checkpoint_cfg,
                gradient=self._config.trainer.gradient,
            ),
            deps=TrainerEngineDependencies(
                model_pipeline=self._state.model_pipeline,
                dataloader=train_dataloader,
                device=self._device,
                output_dir=None,
                tb_logger=self._state.tb_logger,
                run_config=self._config,
            ),
        )

    def client_update(
        self, global_params: OrderedDict[str, torch.Tensor]
    ) -> FLClientOutput:
        num_train_samples = len(self._state.data_pipeline.dataset.train)
        logger.debug(
            f"[Client {self._client_id}] update starting: loaded global params, "
            f"{num_train_samples} local train samples, "
            f"{self._config.trainer.max_epochs} local epoch(s), device={self._device}"
        )

        self._state.model_pipeline._model.load_state_dict(global_params, strict=True)

        engine = self._build_train_engine()
        state = engine.run(checkpoint_path=None)

        # engine.run() moved the model onto self._device (GPU) to train it; move it back
        # to CPU now that this client's turn is over so it doesn't keep occupying GPU
        # memory until (possibly many rounds later) it's selected again.
        self._state.model_pipeline.ops.to_device(torch.device("cpu"))

        params = OrderedDict(
            (k, v.detach().cpu().clone())
            for k, v in self._state.model_pipeline._model.state_dict().items()
        )
        metrics = dict(state.metrics) if state is not None else None
        logger.debug(
            f"[Client {self._client_id}] update finished; offloaded model to CPU; "
            f"metrics={metrics}"
        )
        return FLClientOutput(params=params, metrics=metrics)
