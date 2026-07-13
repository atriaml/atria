from __future__ import annotations

import logging
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa
from atria_logger import get_logger
from atria_ml.data_pipeline._data_pipeline import DataPipeline
from atria_ml.task_pipelines._trainer import Trainer, TrainerState
from atria_ml.task_pipelines._utilities import _initialize_torch
from atria_ml.training.utilities.torch_utils import _initialize_torch
from atria_models.core.model_pipelines._model_pipeline import ModelPipeline

from atria_prv.feamdp._engines._feam_dp_trainer_engine import (
    FeAmDPTrainerEngine,
    FeAmDPTrainerEngineConfig,
    FeAmDPTrainerEngineDependencies,
)
from atria_prv.feamdp.configs import FeAmDPClientTrainingTaskConfig

if TYPE_CHECKING:
    from ignite.handlers import TensorboardLogger

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
class FeAmDPClientTrainerState(TrainerState):
    client_id: int | None = None


@dataclass
class FeAmDPClientOutput:
    grads: OrderedDict[str, torch.Tensor]
    metrics: dict | None = None
    num_samples: int = 0


class FeAmDPClientTrainer(Trainer):
    config: FeAmDPClientTrainingTaskConfig

    def __init__(
        self,
        config: FeAmDPClientTrainingTaskConfig,
        model_pipeline: ModelPipeline | None = None,
    ) -> None:
        self._config = config

        # this is useful in case the client is running in the same seqeuntial process
        # this helps us to avoid creating copies of the model and save memory
        self._model_pipeline = model_pipeline
        self._state: FeAmDPClientTrainerState = self._build()

    def _initialize_runtime(self) -> None:
        import ignite.distributed as idist
        import torch

        # initialize training
        _initialize_torch(
            seed=self._config.env.seed + self._config.client_id + 1,
            deterministic=self._config.env.deterministic,
        )

        # initialize torch device (cpu or gpu)
        if torch.cuda.is_available():
            self._device = idist.device()
        else:
            self._device = "cpu"

        logger.info(
            f"Seed set to {self._config.env.seed + self._config.client_id + 1} and device to {self._device} on client {self._config.client_id}."
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
        logger.info(f"[Client {self._config.client_id}] Dataset:\n{dataset}")

        # build model pipeline
        with suppress_logging():
            if self._model_pipeline is not None:
                model_pipeline = self._model_pipeline
            else:
                model_pipeline = self._config.model_pipeline.build(labels=labels)

            # build data pipeline
            data_pipeline = DataPipeline(dataset=dataset)

        return FeAmDPClientTrainerState(
            data_pipeline=data_pipeline,
            model_pipeline=model_pipeline,
            tb_logger=tb_logger,
            client_id=self._config.client_id,
        )

    def _build_train_engine(self) -> FeAmDPTrainerEngine:
        train_dataloader = self._state.data_pipeline.train_dataloader(
            batch_size=self._config.data.train_batch_size,
            num_workers=self._config.data.num_workers,
            pin_memory=self._config.data.pin_memory,
        )

        return FeAmDPTrainerEngine(
            config=FeAmDPTrainerEngineConfig(
                max_epochs=1,
                logging=self._config.logging,
                test_run=self._config.test_run,
                dp_config=self._config.dp_config,
            ),
            deps=FeAmDPTrainerEngineDependencies(
                model_pipeline=self._state.model_pipeline,
                dataloader=train_dataloader,
                device=torch.device(self._device),
                output_dir=None,
                tb_logger=self._state.tb_logger,
            ),
        )

    # def train(
    #     self, global_params: OrderedDict[str, torch.Tensor]
    # ) -> FeAmDPClientOutput:
    #     # num_train_samples = len(self._state.data_pipeline.dataset.train)
    #     # logger.debug(
    #     #     f"[Client {self._config.client_id}] update starting: loaded global params, "
    #     #     f"{num_train_samples} local train samples, "
    #     # )

    #     # # for sanity check lets print first few values of first 10 old - grad of the model
    #     # for idx, (name, param) in enumerate(global_params.items()):
    #     #     if idx >= 10:
    #     #         break
    #     #     logger.info(
    #     #         f"[Client {self._config.client_id}] model param {name}: "
    #     #         f"{param.detach().cpu().numpy().flatten()[:10]}"
    #     #     )
    #     self._state.model_pipeline._model.load_state_dict(global_params, strict=True)

    #     # old_state_dict = OrderedDict(
    #     #     (k, v.detach().cpu().clone())
    #     #     for k, v in self._state.model_pipeline._model.state_dict().items()
    #     # )

    #     engine = self._build_train_engine()
    #     state = engine.run()
    #     grads = OrderedDict(
    #         (name, param.grad.detach().cpu().clone())
    #         for name, param in self._state.model_pipeline._model.named_parameters()
    #         if param.grad is not None
    #     )
    #     self._state.model_pipeline.ops.to_device(torch.device("cpu"))
    #     metrics = dict(state.metrics) if state is not None else None
    #     # logger.debug(
    #     #     f"[Client {self._config.client_id}] update finished; offloaded model to CPU; "
    #     #     f"metrics={metrics}"
    #     # )

    #     # new_state_dict = self._state.model_pipeline._model.state_dict()

    #     # # sanity check: old_param - grad should equal new_param (lr=1.0, momentum=0.0)
    #     # for idx, (name, old_param) in enumerate(old_state_dict.items()):
    #     #     if name not in grads:
    #     #         raise RuntimeError(f"Gradient not found for param = {name}")

    #     #     new_param = new_state_dict[name].detach().cpu()
    #     #     expected_new = old_param - grads[name]

    #     #     is_close = torch.allclose(expected_new, new_param)
    #     #     if idx < 10:
    #     #         logger.debug(
    #     #             f"[Client {self._config.client_id}] sanity check {name}: "
    #     #             f"old-grad==new? {is_close}"
    #     #         )
    #     #     if not is_close:
    #     #         max_diff = (expected_new - new_param).abs().max().item()
    #     #         logger.warning(
    #     #             f"[Client {self._config.client_id}] SANITY CHECK FAILED for {name}: "
    #     #             f"max diff = {max_diff}"
    #     #         )

    #     # # for sanity check lets print first few values of first 10 old - grad of the model
    #     # for idx, (name, param) in enumerate(
    #     #     self._state.model_pipeline._model.state_dict().items()
    #     # ):
    #     #     if idx >= 10:
    #     #         break
    #     #     logger.debug(
    #     #         f"[Client {self._config.client_id}] model param {name}: "
    #     #         f"{param.detach().cpu().numpy().flatten()[:10]}"
    #     #     )

    #     torch.cuda.empty_cache()
    #     return FeAmDPClientOutput(grads=grads, metrics=metrics, num_samples=1)

    def train(
        self, global_params: OrderedDict[str, torch.Tensor]
    ) -> FeAmDPClientOutput:
        self._state.model_pipeline._model.load_state_dict(global_params, strict=True)
        engine = self._build_train_engine()
        state = engine.run()
        grads = OrderedDict(
            (name, param.grad.detach().cpu().clone())
            for name, param in self._state.model_pipeline._model.named_parameters()
            if param.grad is not None
        )
        self._state.model_pipeline.ops.to_device(torch.device("cpu"))
        metrics = dict(state.metrics) if state is not None else None
        torch.cuda.empty_cache()
        return FeAmDPClientOutput(grads=grads, metrics=metrics, num_samples=1)
