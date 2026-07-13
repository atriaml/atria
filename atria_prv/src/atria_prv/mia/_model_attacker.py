from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa
from atria_logger import enable_file_logging
from atria_logger._api import get_logger
from atria_ml.task_pipelines._utilities import (
    MODEL_PIPELINE_CHECKPOINT_KEY,
    _get_env_info,
    _initialize_torch,
)
from atria_models.core.model_pipelines._model_pipeline import ModelPipeline
from omegaconf import OmegaConf

from atria_prv.mia._attack import MembershipInferenceAttack
from atria_prv.mia._attack_data_pipeline import AttackDataPipeline
from atria_prv.mia._extraction_engine import SignalExtractionEngine
from atria_prv.mia.configs import MembershipInferenceTaskConfig

if TYPE_CHECKING:
    from ignite.handlers import TensorboardLogger

logger = get_logger(__name__)


@dataclass
class ModelAttackerState:
    data_pipeline: AttackDataPipeline
    model_pipeline: ModelPipeline
    tb_logger: TensorboardLogger | None = None


class ModelAttacker:
    def __init__(
        self, config: MembershipInferenceTaskConfig, local_rank: int = 0
    ) -> None:
        self._config = config
        self._state: ModelAttackerState = self._build(local_rank=local_rank)

    # ------------------------------------------------------------------ setup
    def _initialize_runtime(self, local_rank: int) -> None:
        import ignite.distributed as idist
        import torch

        env_info = _get_env_info()
        _initialize_torch(
            seed=self._config.env.seed, deterministic=self._config.env.deterministic
        )

        if torch.cuda.is_available():
            self._device = idist.device()
        else:
            self._device = "cpu"

        logger.info(
            f"Environment info:\n{yaml.dump(OmegaConf.to_container(OmegaConf.create(env_info)), indent=4)}"
        )
        dumped = self._config.model_dump()
        logger.info(
            f"Run configuration:\n{yaml.dump(OmegaConf.to_container(OmegaConf.create(dumped)), indent=4)}"
        )
        logger.info(f"Seed set to {self._config.env.seed} on device: {self._device}")

    def _setup_logging(self) -> TensorboardLogger | None:
        import ignite.distributed as idist
        from ignite.handlers import TensorboardLogger

        if idist.get_rank() == 0:
            log_dir = Path(self._config.env.run_dir) / "tensorboard"
            log_dir.mkdir(parents=True, exist_ok=True)
            tb_logger = TensorboardLogger(log_dir=log_dir)
            enable_file_logging(str(Path(self._config.env.run_dir) / "attack.log"))
        else:
            tb_logger = None
        return tb_logger

    def _build(self, local_rank: int) -> ModelAttackerState:
        self._initialize_runtime(local_rank=local_rank)
        tb_logger = self._setup_logging()

        train_transform = self._config.model_pipeline.train_transform
        eval_transform = self._config.model_pipeline.eval_transform

        dataset = self._config.data.build_dataset()
        dataset.apply_transforms(
            train_transform=train_transform, eval_transform=eval_transform
        )
        labels = dataset.metadata.dataset_labels
        logger.info(f"Dataset:\n{dataset}")

        model_pipeline = self._config.model_pipeline.build(labels=labels)
        logger.info(model_pipeline.ops.summarize())

        # load the trained target model checkpoint (the model under attack)
        import torch
        from ignite.handlers.checkpoint import Checkpoint

        if not Path(self._config.target_checkpoint).exists():
            raise FileNotFoundError(
                f"Target checkpoint not found: {self._config.target_checkpoint}"
            )
        logger.info(f"Loading target checkpoint: {self._config.target_checkpoint}")
        checkpoint = torch.load(
            self._config.target_checkpoint, map_location="cpu", weights_only=False
        )
        Checkpoint.load_objects(
            to_load={MODEL_PIPELINE_CHECKPOINT_KEY: model_pipeline},
            checkpoint=checkpoint,
            strict=True,
        )

        data_pipeline = AttackDataPipeline(
            dataset=dataset,
            attack_train_ratio=self._config.attack_config.attack_train_ratio,
            batch_size=self._config.data.eval_batch_size,
            num_workers=self._config.data.num_workers,
            pin_memory=self._config.data.pin_memory,
        )
        data_pipeline.summarize()

        return ModelAttackerState(
            data_pipeline=data_pipeline,
            model_pipeline=model_pipeline,
            tb_logger=tb_logger,
        )

    # ------------------------------------------------------------------ run
    def run(self) -> dict:
        dataloaders = self._state.data_pipeline.dataloaders()
        extractor = SignalExtractionEngine(self._state.model_pipeline, self._device)

        x_members, y_members, _ = extractor.extract(dataloaders.members_train)
        x_nonmembers, y_nonmembers, _ = extractor.extract(nonmember_loader)

        # 3) run the attack
        num_labels = len(self._state.model_pipeline._labels.ser)
        attack = MembershipInferenceAttack(cfg)
        results = attack.run(
            num_labels=num_labels,
            x_members=x_members,
            y_members=y_members,
            x_nonmembers=x_nonmembers,
            y_nonmembers=y_nonmembers,
        )

        logger.info(
            f"Membership inference attack results:\n"
            f"{yaml.dump(results, indent=4, default_flow_style=False)}"
        )
        self._config.dump_metrics_file(data=results)
        return results
