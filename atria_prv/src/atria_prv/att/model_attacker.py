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

from atria_prv.att._attacks._attack import MembershipInferenceAttack
from atria_prv.att._data._attack_data_pipeline import AttackDataPipeline
from atria_prv.att._features._extraction_engine import SignalExtractionEngine
from atria_prv.att._features._extractor import TokenSignalExtractor
from atria_prv.att._features._factory import build_feature_extractor
from atria_prv.att.configs import MembershipInferenceTaskConfig

if TYPE_CHECKING:
    from ignite.handlers import TensorboardLogger

logger = get_logger(__name__)


@dataclass
class ModelAttackerState:
    data_pipeline: AttackDataPipeline
    model_pipeline: ModelPipeline
    extractor: TokenSignalExtractor
    tb_logger: TensorboardLogger | None = None


class ModelAttacker:
    _SPLIT_NAMES = (
        "members_train",
        "non_members_train",
        "members_test",
        "non_members_test",
    )

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

        # fail fast on an unsupported pipeline type, before paying for checkpoint I/O
        extractor = build_feature_extractor(model_pipeline, labels)

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
            extractor=extractor,
            tb_logger=tb_logger,
        )

    # ------------------------------------------------------------------ extraction cache
    def _cache_dir(self) -> Path:
        import hashlib
        import json
        from dataclasses import asdict

        ckpt = Path(self._config.target_checkpoint)
        key = {
            "checkpoint": str(ckpt.resolve()),
            "checkpoint_mtime": ckpt.stat().st_mtime if ckpt.exists() else None,
            "dataset": self._config.env.dataset_name,
            "attack_train_ratio": self._config.attack_config.attack_train_ratio,
            "eval_batch_size": self._config.data.eval_batch_size,
            "balanced": self._state.data_pipeline._balanced,
            "data_seed": self._state.data_pipeline._seed,
            "signals": self._state.extractor.signal_names,
            "agg_config": asdict(self._state.extractor.config),
        }
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[
            :16
        ]
        return Path(self._config.env.run_dir) / "attack_cache" / f"features_{digest}"

    def _extract_features(self, loaders) -> dict:
        """Extract (or load from cache) the per-sample feature DataFrames for the four splits."""
        import pandas as pd

        cache_dir = self._cache_dir()
        cache_files = {
            split: cache_dir / f"{split}.parquet" for split in self._SPLIT_NAMES
        }
        if all(path.exists() for path in cache_files.values()):
            logger.info(f"Loading cached extracted features from {cache_dir}")
            return {split: pd.read_parquet(path) for split, path in cache_files.items()}

        engine = SignalExtractionEngine(
            self._state.model_pipeline, self._device, self._state.extractor
        )
        features = {
            "members_train": engine.extract(loaders.members_train),
            "non_members_train": engine.extract(loaders.non_members_train),
            "members_test": engine.extract(loaders.members_test),
            "non_members_test": engine.extract(loaders.non_members_test),
        }
        cache_dir.mkdir(parents=True, exist_ok=True)
        for split, df in features.items():
            df.to_parquet(cache_files[split])
        logger.info(f"Saved extracted features to {cache_dir}")
        return features

    # ------------------------------------------------------------------ run
    def run(self) -> dict:
        loaders = self._state.data_pipeline.dataloaders()
        features = self._extract_features(loaders)

        # debug: plot the member vs. non-member train mean-loss distributions
        if self._config.env.run_dir is not None:
            from atria_prv.att._utils._plots import save_loss_distributions

            dist_path = Path(self._config.env.run_dir) / "loss_distributions_train.png"
            save_loss_distributions(
                features["members_train"]["loss__all__mean"].to_numpy(),
                features["non_members_train"]["loss__all__mean"].to_numpy(),
                dist_path,
            )
            logger.info(f"Saved train-loss distributions to {dist_path}")

        results = MembershipInferenceAttack(self._config.attack_config).run(
            num_labels=len(self._state.model_pipeline._labels.ser),
            features_members_train=features["members_train"],
            features_nonmembers_train=features["non_members_train"],
            features_members_test=features["members_test"],
            features_nonmembers_test=features["non_members_test"],
        )

        # draw + save the ROC curve
        roc = results.get("roc_curve")
        if roc is not None and self._config.env.run_dir is not None:
            from atria_prv.att._utils._plots import save_roc_curve

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
