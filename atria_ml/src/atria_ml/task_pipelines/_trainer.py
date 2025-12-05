from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from atria_datasets.api.datasets import load_dataset_config
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa
from atria_logger._api import get_logger
from atria_models.api.models import load_model_pipeline_config
from atria_models.core.model_pipelines._common import ModelConfig
from omegaconf import OmegaConf

from atria_ml.data_pipeline._data_pipeline import DataPipeline
from atria_ml.task_pipelines._utilities import _get_env_info, _initialize_torch
from atria_ml.task_pipelines.configs._base import (
    DataConfig,
    RunnerConfig,
    RuntimeEnvConfig,
)

if TYPE_CHECKING:
    from ignite.handlers import TensorboardLogger

logger = get_logger(__name__)


class Trainer:
    def __init__(self, config: RunnerConfig) -> None:
        self._config = config

    def _initialize_runtime(self, local_rank: int) -> None:
        # Log system information
        env_info = _get_env_info()

        # initialize training
        _initialize_torch(
            seed=self._config.env.seed, deterministic=self._config.env.deterministic
        )

        # initialize torch device (cpu or gpu)
        self._device = local_rank

        # log env info and run configuration
        logger.info(
            f"Environment info:\n{yaml.dump(OmegaConf.to_container(OmegaConf.create(env_info)), indent=4)}"
        )
        logger.info(
            f"Run configuration:\n{yaml.dump(OmegaConf.to_container(OmegaConf.create(self._config.model_dump())), indent=4)}"
        )
        logger.info(f"Seed set to {self._config.env.seed} on device: {self._device}")

    def _setup_logging(self) -> TensorboardLogger:
        import ignite.distributed as idist
        from ignite.handlers import TensorboardLogger

        if idist.get_rank() == 0:
            log_dir = Path(self._config.env.output_dir) / "tensorboard"
            log_dir.mkdir(parents=True, exist_ok=True)
            tb_logger = TensorboardLogger(log_dir=log_dir)
        else:
            tb_logger = None
        return tb_logger

    # def _build_training_engine(self) -> None:
    #     if self._training_engine is not None:
    #         logger.info("Setting up training engine")
    #         self._training_engine: AtriaEngine = self._training_engine.build(
    #             run_config=self._run_config,
    #             output_dir=self._output_dir,
    #             model_pipeline=self._model_pipeline,
    #             dataloader=train_dataloader,
    #             device=self._device,
    #             tb_logger=self._tb_logger,
    #             validation_engine=self._validation_engine,
    #             visualization_engine=self._visualization_engine,
    #         )

    #     if self._validation_engine is not None:
    #         logger.info("Setting up validation engine")
    #         self._validation_engine: AtriaEngine = self._validation_engine.build(
    #             output_dir=self._output_dir,
    #             model_pipeline=self._model_pipeline,
    #             dataloader=validation_dataloader,
    #             device=self._device,
    #             tb_logger=self._tb_logger,
    #         )

    #     if self._visualization_engine is not None:
    #         logger.info("Setting up visualization engine")
    #         self._visualization_engine: AtriaEngine = self._visualization_engine.build(
    #             output_dir=self._output_dir,
    #             model_pipeline=self._model_pipeline,
    #             dataloader=visualization_dataloader,
    #             device=self._device,
    #             tb_logger=self._tb_logger,
    #         )

    def _build_test_engine(self) -> None:
        if self._test_engine is not None:
            self._test_engine: AtriaEngine = self._test_engine.build(
                output_dir=self._output_dir,
                model_pipeline=self._model_pipeline,
                dataloader=test_dataloader,
                device=self._device,
                tb_logger=self._tb_logger,
            )

    def build(self, local_rank: int) -> None:
        self._initialize_runtime(local_rank=local_rank)
        self._setup_logging()

        # build dataset
        dataset = self._config.data.build_dataset()

        # load labels
        labels = dataset.metadata.dataset_labels

        # build model pipeline
        model_pipeline = self._config.model_pipeline.build(labels=labels)

        # log model pipeline
        logger.info(model_pipeline)

        # get model transforms
        train_transform = model_pipeline.config.train_transform
        eval_transform = model_pipeline.config.eval_transform
        if dataset.train is not None:
            dataset.train.output_transform = train_transform
        if dataset.validation is not None:
            dataset.validation.output_transform = eval_transform
        if dataset.test is not None:
            dataset.test.output_transform = eval_transform

        # build data pipeline
        data_pipeline = DataPipeline(dataset=dataset)

        for batch in data_pipeline.dataloader(
            split="train", batch_size=4, num_workers=0
        ):
            logger.info(f"Train batch: {batch}")
            break

        # build test engine
        test_engine =
    # def train(self) -> None:
    #     if self._do_train:
    #         self._build_training_engine()
    #         self._training_engine.run(resume_checkpoint=self._resume_checkpoint)

    # def test(self) -> None:
    #     if self._do_test:
    #         self._build_test_engine()
    #         self._test_engine.run(test_checkpoint=self._test_checkpoint)

    # def run(self) -> None:
    #     self.train()
    #     self.test()


# config = TrainerConfig(
#     env=RuntimeEnvConfig(
#         run_name="test_trainer",
#         output_dir="outputs/test_trainer",
#         seed=42,
#         deterministic=True,
#     ),
#     model_pipeline=load_model_pipeline_config(
#         "image_classification",
#         model=ModelConfig(model_name_or_path="resnet18"),
#         # train_transform=load_transform("image_processor"),
#         # eval_transform=load_transform("image_processor"),
#     ),
#     data=DataConfig(dataset_config=load_dataset_config("cifar10/1k")),
# )
config = RunnerConfig(
    env=RuntimeEnvConfig(
        run_name="test_trainer",
        output_dir="outputs/test_trainer",
        seed=42,
        deterministic=True,
    ),
    model_pipeline=load_model_pipeline_config(
        "image_classification",
        model=ModelConfig(model_name_or_path="resnet18"),
        # train_transform=load_transform("image_processor"),
        # eval_transform=load_transform("image_processor"),
    ),
    data=DataConfig(
        dataset_config=load_dataset_config("cifar10/1k"),
        data_dir="data_dir/",
        num_workers=0,
    ),
)

Trainer(config=config).build(local_rank=0)
