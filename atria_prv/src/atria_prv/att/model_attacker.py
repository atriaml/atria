from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import yaml
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa
from atria_logger import enable_file_logging
from atria_logger._api import get_logger
from atria_ml.task_pipelines._utilities import _get_env_info, _initialize_torch
from atria_ml.training.engines._test_engine import (
    TestEngine,
    TestEngineConfig,
    TestEngineDependencies,
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
    labels: object
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
        self._run_dir = self._compute_run_dir()
        self._state: ModelAttackerState = self._build(local_rank=local_rank)

        if self._config.feature_set == "A":
            # Feature columns the attack model is trained/evaluated on (shared by both modes).
            self._feature_columns = ["loss__all__mean", "prob__all__mean"]
        elif self._config.feature_set == "B":
            # Feature columns the attack model is trained/evaluated on (shared by both modes).
            self._feature_columns = [
                "loss__all__mean",
                "loss__entity__mean",
                "loss__span_start__mean",
                "loss__span_cont__mean",
                "loss__other__mean",
                "prob__all__mean",
                "prob__entity__mean",
                "prob__span_start__mean",
                "prob__span_cont__mean",
                "prob__other__mean",
            ]
        else:
            raise RuntimeError()

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
            log_dir = Path(self._run_dir) / "tensorboard"
            log_dir.mkdir(parents=True, exist_ok=True)
            tb_logger = TensorboardLogger(log_dir=log_dir)
            enable_file_logging(str(Path(self._run_dir) / "attack.log"))
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
        extractor = build_feature_extractor(model_pipeline)

        # load the trained target model checkpoint (the model under attack)

        if not Path(self._config.target_checkpoint).exists():
            raise FileNotFoundError(
                f"Target checkpoint not found: {self._config.target_checkpoint}"
            )
        data_pipeline = AttackDataPipeline(
            dataset=dataset,
            attack_train_ratio=self._config.attack_config.attack_train_ratio,
            batch_size=self._config.data.eval_batch_size,
            num_workers=self._config.data.num_workers,
            pin_memory=self._config.data.pin_memory,
            seed_train_val=42,
            seed_train_reshuffle=self._config.env.seed,
        )
        data_pipeline.summarize()
        return ModelAttackerState(
            data_pipeline=data_pipeline,
            model_pipeline=model_pipeline,
            extractor=extractor,
            labels=labels,
            tb_logger=tb_logger,
        )

    # ------------------------------------------------------------------ attack directory
    def _compute_run_dir(self) -> Path:
        import hashlib
        import json

        ckpt = Path(self._config.target_checkpoint)
        # find experiment base path
        checkpoint_type = ckpt.parent.parent.parent.parent.name
        key = {
            "checkpoint_type": checkpoint_type,
            "checkpoint": str(ckpt.resolve()),
            "checkpoint_mtime": ckpt.stat().st_mtime if ckpt.exists() else None,
            **self._config.attack_config.model_dump(),
            "shadow": self._config.shadow_config.model_dump(),
        }
        mode = "shadow" if self._config.shadow_config.enabled else "direct"
        digest = hashlib.sha256(
            json.dumps(key, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        return (
            Path(self._config.env.run_dir)
            / f"checkpoint_type={checkpoint_type}_mode={mode}_att_m={self._config.attack_config.attack_model_type}_{digest}"
        )

    def _build_test_engine(self) -> TestEngine:
        import torch

        test_dataloader = self._state.data_pipeline.test_dataloader(
            batch_size=self._config.data.eval_batch_size,
            num_workers=self._config.data.num_workers,
            pin_memory=self._config.data.pin_memory,
        )
        return TestEngine(
            config=TestEngineConfig(
                logging=self._config.logging, test_run=self._config.test_run
            ),
            deps=TestEngineDependencies(
                model_pipeline=self._state.model_pipeline,
                dataloader=test_dataloader,
                device=torch.device(self._device),
                output_dir=self._config.env.run_dir,
                tb_logger=self._state.tb_logger,
            ),
        )

    def _extract_features(self, loaders) -> dict:
        """Extract (or load from cache) the per-sample feature DataFrames for the four splits."""
        import pandas as pd

        features_dir = self._run_dir / "features"
        cache_files = {
            split: features_dir / f"{split}.parquet" for split in self._SPLIT_NAMES
        }
        if all(path.exists() for path in cache_files.values()):
            logger.info(f"Loading cached extracted features from {features_dir}")
            return {split: pd.read_parquet(path) for split, path in cache_files.items()}

        logger.info(f"Loading target checkpoint: {self._config.target_checkpoint}")
        checkpoint = torch.load(
            self._config.target_checkpoint, map_location="cpu", weights_only=False
        )
        self._state.model_pipeline._model.load_state_dict(
            checkpoint["model_pipeline"]["model"]
        )
        # Checkpoint.load_objects(
        #     to_load={MODEL_PIPELINE_CHECKPOINT_KEY: model_pipeline},
        #     checkpoint=checkpoint,
        #     strict=True,
        # )
        engine = SignalExtractionEngine(
            self._state.model_pipeline, self._device, self._state.extractor
        )
        features = {
            "members_train": engine.extract(loaders.members_train),
            "non_members_train": engine.extract(loaders.non_members_train),
            "members_test": engine.extract(loaders.members_test),
            "non_members_test": engine.extract(loaders.non_members_test),
        }
        features_dir.mkdir(parents=True, exist_ok=True)
        for split, df in features.items():
            df.to_parquet(cache_files[split])
        logger.info(f"Saved extracted features to {features_dir}")
        return features

    # ------------------------------------------------------------------ shadow mode
    def _extract_with_model(self, model_pipeline, dataloader):
        """Run one model over one loader and return its per-sample feature DataFrame."""
        engine = SignalExtractionEngine(
            model_pipeline, self._device, self._state.extractor
        )
        return engine.extract(dataloader)

    def _build_shadow_features(self) -> dict:
        """Shadow-model mode: attack-train features come from shadow models, attack-test
        features come from the target model on its full member / non-member sets."""
        import pandas as pd

        from atria_prv.att._shadow._shadow_trainer import ShadowModelTrainer

        sc = self._config.shadow_config
        dp = self._state.data_pipeline

        # --- attack-train: shadow models' in (member) / out (non-member) features ---
        shadow_dir = self._run_dir / "shadow_features"
        member_cache = shadow_dir / "shadow_members.parquet"
        nonmember_cache = shadow_dir / "shadow_nonmembers.parquet"
        if member_cache.exists() and nonmember_cache.exists():
            logger.info(f"Loading cached shadow features from {shadow_dir}")
            shadow_members = pd.read_parquet(member_cache)
            shadow_nonmembers = pd.read_parquet(nonmember_cache)
        else:
            trainer = ShadowModelTrainer(
                config=self._config,
                labels=self._state.labels,
                device=self._device,
                tb_logger=self._state.tb_logger,
            )
            splits = dp.shadow_splits(
                num_models=sc.num_shadow_models, in_ratio=sc.in_ratio, seed=sc.seed
            )
            member_frames, nonmember_frames = [], []
            for i, (in_split, out_split) in enumerate(splits):
                model_dir = self._run_dir / "shadows" / f"shadow_{i}"
                shadow_pipeline = trainer.train(in_split, model_dir)
                logger.info(f"Extracting shadow {i} in/out features")
                member_frames.append(
                    self._extract_with_model(shadow_pipeline, dp.loader(in_split))
                )
                nonmember_frames.append(
                    self._extract_with_model(shadow_pipeline, dp.loader(out_split))
                )
            shadow_members = pd.concat(member_frames, ignore_index=True)
            shadow_nonmembers = pd.concat(nonmember_frames, ignore_index=True)
            shadow_dir.mkdir(parents=True, exist_ok=True)
            shadow_members.to_parquet(member_cache)
            shadow_nonmembers.to_parquet(nonmember_cache)

        # --- attack-test: target model on full members (train) / non-members (test) ---
        features_dir = self._run_dir / "features"
        tm_cache = features_dir / "target_members.parquet"
        tn_cache = features_dir / "target_non_members.parquet"
        if tm_cache.exists() and tn_cache.exists():
            logger.info(f"Loading cached target features from {features_dir}")
            target_members = pd.read_parquet(tm_cache)
            target_nonmembers = pd.read_parquet(tn_cache)
        else:
            members_split, nonmembers_split = dp.target_eval_splits()
            target_members = self._extract_with_model(
                self._state.model_pipeline, dp.loader(members_split)
            )
            target_nonmembers = self._extract_with_model(
                self._state.model_pipeline, dp.loader(nonmembers_split)
            )
            features_dir.mkdir(parents=True, exist_ok=True)
            target_members.to_parquet(tm_cache)
            target_nonmembers.to_parquet(tn_cache)

        return {
            "members_train": shadow_members,
            "non_members_train": shadow_nonmembers,
            "members_test": target_members,
            "non_members_test": target_nonmembers,
        }

    # ------------------------------------------------------------------ run
    def _run_attack(self, features: dict) -> dict:
        """Fit + evaluate the attack model over the four feature splits and report."""
        # # debug: plot the member vs. non-member train distributions for all features
        # dist_path = self._run_dir / "feature_distributions_train.png"
        # written = save_feature_distributions(
        #     features["members_train"], features["non_members_train"], dist_path
        # )
        # logger.info(f"Saved train feature distributions to {written}")

        print("self._feature_columns", self._feature_columns)
        results = MembershipInferenceAttack(self._config.attack_config).run(
            features_members_train=features["members_train"],
            features_nonmembers_train=features["non_members_train"],
            features_members_test=features["members_test"],
            features_nonmembers_test=features["non_members_test"],
            feature_columns=self._feature_columns,
            output_dir=self._run_dir / "attack_results",
            cache_key=self._config.attack_config.attack_model_type,
        )

        # draw + save the ROC curve
        roc = results.get("roc_curve")
        if roc is not None:
            from atria_prv.att._utils._plots import save_roc_curve

            roc_path = self._run_dir / "roc_curve.png"
            save_roc_curve(roc["fpr"], roc["tpr"], results["auc"], roc_path)
            logger.info(f"Saved ROC curve to {roc_path}")

        log_results = {k: v for k, v in results.items() if k != "roc_curve"}
        from rich.pretty import pprint

        pprint({k: v for k, v in log_results.items() if k not in ["_raw"]})
        self._config.dump_metrics_file(data=results)
        return results

    def run(self) -> dict:
        do_test = False
        if do_test:
            logger.info(f"Loading target checkpoint: {self._config.target_checkpoint}")
            checkpoint = torch.load(
                self._config.target_checkpoint, map_location="cpu", weights_only=False
            )
            self._state.model_pipeline._model.load_state_dict(
                checkpoint["model_pipeline"]["model"]
            )
            test_engine = self._build_test_engine()
            test_engine.run()

        if self._config.shadow_config.enabled:
            logger.info(
                f"Running shadow-model attack with "
                f"{self._config.shadow_config.num_shadow_models} shadow models."
            )
            features = self._build_shadow_features()
        else:
            loaders = self._state.data_pipeline.dataloaders()
            features = self._extract_features(loaders)
        return self._run_attack(features)
