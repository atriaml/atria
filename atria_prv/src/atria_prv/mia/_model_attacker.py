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

    # ------------------------------------------------------------------ extraction cache
    def _cache_path(self) -> Path:
        import hashlib
        import json

        ckpt = Path(self._config.target_checkpoint)
        key = {
            "checkpoint": str(ckpt.resolve()),
            "checkpoint_mtime": ckpt.stat().st_mtime if ckpt.exists() else None,
            "dataset": self._config.env.dataset_name,
            "attack_train_ratio": self._config.attack_config.attack_train_ratio,
            "eval_batch_size": self._config.data.eval_batch_size,
            "balanced": self._state.data_pipeline._balanced,
            "data_seed": self._state.data_pipeline._seed,
        }
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[
            :16
        ]
        return Path(self._config.env.run_dir) / "attack_cache" / f"losses_{digest}.npz"

    def _extract_losses(self, loaders) -> dict:
        """Extract (or load from cache) the per-document losses for the four splits."""
        import numpy as np

        cache_path = self._cache_path()
        if cache_path.exists():
            logger.info(f"Loading cached extracted losses from {cache_path}")
            data = np.load(cache_path)
            return {k: data[k] for k in data.files}

        extractor = SignalExtractionEngine(self._state.model_pipeline, self._device)
        losses = {
            "members_train": extractor.extract(loaders.members_train),
            "non_members_train": extractor.extract(loaders.non_members_train),
            "members_test": extractor.extract(loaders.members_test),
            "non_members_test": extractor.extract(loaders.non_members_test),
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **losses)
        logger.info(f"Saved extracted losses to {cache_path}")
        return losses

    # ------------------------------------------------------------------ run
    def run(self) -> dict:
        loaders = self._state.data_pipeline.dataloaders()
        losses = self._extract_losses(loaders)

        # debug: plot the member vs. non-member train-loss distributions
        if self._config.env.run_dir is not None:
            from atria_prv.mia._plots import save_loss_distributions

            dist_path = Path(self._config.env.run_dir) / "loss_distributions_train.png"
            save_loss_distributions(
                losses["members_train"], losses["non_members_train"], dist_path
            )
            logger.info(f"Saved train-loss distributions to {dist_path}")

        results = MembershipInferenceAttack(self._config.attack_config).run(
            num_labels=len(self._state.model_pipeline._labels.ser),
            loss_members_train=losses["members_train"],
            loss_nonmembers_train=losses["non_members_train"],
            loss_members_test=losses["members_test"],
            loss_nonmembers_test=losses["non_members_test"],
        )

        # draw + save the ROC curve
        roc = results.get("roc_curve")
        if roc is not None and self._config.env.run_dir is not None:
            from atria_prv.mia._plots import save_roc_curve

            roc_path = Path(self._config.env.run_dir) / "roc_curve.png"
            save_roc_curve(roc["fpr"], roc["tpr"], results["auc"], roc_path)
            logger.info(f"Saved ROC curve to {roc_path}")

        log_results = {k: v for k, v in results.items() if k != "roc_curve"}
        logger.info(
            f"Membership inference attack results:\n"
            f"{yaml.dump(log_results, indent=4, default_flow_style=False)}"
        )
        self._config.dump_metrics_file(data=results)
        return results
