from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from atria_datasets.core.dataset._datasets import Dataset
from atria_datasets.core.dataset._exceptions import SplitNotFoundError
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa
from atria_logger._api import get_logger
from atria_ml.data_pipeline._data_pipeline import DataPipeline
from atria_ml.task_pipelines._utilities import _get_env_info, _initialize_torch
from atria_ml.training.engines._test_engine import (
    TestEngine,
    TestEngineConfig,
    TestEngineDependencies,
)
from atria_ml.training.engines.utilities import _format_metrics_for_logging
from omegaconf import OmegaConf
from atria_logger._api import enable_file_logging, get_logger

from atria_insights.configs.explanation_task_config import ExplanationTaskConfig
from atria_insights.engines._explanation_engine import (
    ExplanationEngine,
    ExplanationEngineConfig,
    ExplanationEngineDependencies,
)
from atria_insights.engines._feature_generation_engine import (
    FeatureGenerationEngine,
    FeatureGenerationEngineConfig,
    FeatureGenerationEngineDependencies,
)
from atria_insights.model_pipelines._model_pipeline import ExplainableModelPipeline

if TYPE_CHECKING:
    from ignite.engine import State
    from ignite.handlers import TensorboardLogger

logger = get_logger(__name__)


@dataclass
class ModelExplainerState:
    data_pipeline: DataPipeline
    x_model_pipeline: ExplainableModelPipeline
    tb_logger: TensorboardLogger | None = None

    @property
    def dataset(self) -> Dataset:
        return self.data_pipeline.dataset


class ModelExplainer:
    def __init__(
        self,
        config: ExplanationTaskConfig,
        local_rank: int = 0,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self._config = config
        if checkpoint_path is not None:
            self._checkpoint_path = checkpoint_path
            self._checkpoint_hash = hashlib.sha256(
                str(checkpoint_path).encode("utf-8")
            ).hexdigest()[:8]
            assert Path(self._checkpoint_path).exists(), (
                f"Checkpoint path {checkpoint_path} does not exist."
            )
            self._run_dir = (
                Path(self._config.env.run_dir) / f"checkpoint-{self._checkpoint_hash}"
            )
        else:
            logger.warning(
                "No checkpoint path provided. Using pre-initialized checkpoint for model explainer."
            )
            self._checkpoint_path = None
            self._run_dir = Path(self._config.env.run_dir) / "default_checkpoint"
        self._state: ModelExplainerState = self._build(local_rank=local_rank)

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
            f"Run configuration:\n{yaml.dump(OmegaConf.to_container(OmegaConf.create(self._config.to_dict())), indent=4)}"
        )
        logger.info(f"Seed set to {self._config.env.seed} on device: {self._device}")

    def _setup_logging(self) -> TensorboardLogger | None:
        import ignite.distributed as idist
        from ignite.handlers import TensorboardLogger

        if idist.get_rank() == 0:
            log_dir = Path(self._run_dir) / "tensorboard"
            log_dir.mkdir(parents=True, exist_ok=True)
            tb_logger = TensorboardLogger(log_dir=log_dir)

            explainer_dir = (
                Path(self._run_dir) / self._config.x_model_pipeline.explainer.type.split("/")[-1]
            )
            if not explainer_dir.exists():
                explainer_dir.mkdir(parents=True, exist_ok=True)
            enable_file_logging(str(Path(explainer_dir) / "run.log"))
        else:
            tb_logger = None
        return tb_logger

    def _build_explanation_engine(
        self, total_samples: int | None = None, compute_metrics: bool = True
    ) -> ExplanationEngine:
        import torch

        test_dataloader = self._state.data_pipeline.test_dataloader(
            batch_size=self._config.data.eval_batch_size,
            num_workers=self._config.data.num_workers,
            pin_memory=self._config.data.pin_memory,
            subset_size=total_samples,
        )
        return ExplanationEngine(
            config=ExplanationEngineConfig(
                logging=self._config.logging,
                test_run=self._config.test_run,
                use_fixed_batch_iterator=self._config.use_fixed_batch_iterator,
                enable_outputs_caching=self._config.enable_outputs_caching,
                compute_metrics=compute_metrics,
            ),
            deps=ExplanationEngineDependencies(
                model_pipeline=self._state.x_model_pipeline._model_pipeline,
                x_model_pipeline=self._state.x_model_pipeline,
                dataloader=test_dataloader,
                device=torch.device(self._device),
                output_dir=self._run_dir,
                tb_logger=self._state.tb_logger,
            ),
        )

    def _build_features_generation_engine(self) -> FeatureGenerationEngine:
        import torch

        train_dataloader = self._state.data_pipeline.train_dataloader(
            batch_size=self._config.data.train_batch_size,
            num_workers=self._config.data.num_workers,
            pin_memory=self._config.data.pin_memory,
        )
        return FeatureGenerationEngine(
            config=FeatureGenerationEngineConfig(
                logging=self._config.logging,
                max_features=self._config.max_training_baseline_features,
                use_fixed_batch_iterator=self._config.use_fixed_batch_iterator,
            ),
            deps=FeatureGenerationEngineDependencies(
                model_pipeline=self._state.x_model_pipeline._model_pipeline,
                x_model_pipeline=self._state.x_model_pipeline,
                dataloader=train_dataloader,
                device=torch.device(self._device),
                output_dir=self._run_dir,
                feature_file_name="features.hdf5",
            ),
        )

    def _build(self, local_rank: int = 0) -> ModelExplainerState:
        self._initialize_runtime(local_rank=local_rank)

        # setup logging
        tb_logger = self._setup_logging()

        # build dataset
        dataset = self._config.data.build_dataset()

        # preprocess dataset splits
        if (
            self._config.data.preprocess_train_transform is not None
            and self._config.data.preprocess_eval_transform is not None
        ):
            # process the dataset with a custom transform
            dataset = dataset.process_dataset(
                train_transform=self._config.data.preprocess_train_transform,
                eval_transform=self._config.data.preprocess_eval_transform,
                max_cache_image_size=self._config.data.preprocess_max_cache_image_size,
                num_processes=self._config.data.num_processes,
                processed_data_dir=self._config.data.data_dir,
            )

        # load labels
        labels = dataset.metadata.dataset_labels

        # log dataset info
        logger.info(f"Dataset:\n{dataset}")

        # see if feature baseline generator is attached, then we updates its path
        if self._config.x_model_pipeline.baseline_generator.type == "feature_based":
            # hard coded for now to the path where the features will be stored
            self._config.x_model_pipeline.baseline_generator.unsafe_update(
                features_path=str(Path(self._run_dir) / "features.hdf5")
            )

        # build model pipelines
        x_model_pipeline = self._config.x_model_pipeline.build(
            labels=labels, persist_to_disk=True, cache_dir=self._run_dir
        )

        # log model pipeline
        logger.info(x_model_pipeline.summarize())

        # get model transforms
        train_transform = x_model_pipeline.config.model_pipeline.train_transform
        eval_transform = x_model_pipeline.config.model_pipeline.eval_transform
        try:
            dataset.train.output_transform = train_transform
            dataset.validation.output_transform = eval_transform
            dataset.test.output_transform = eval_transform
        except SplitNotFoundError:
            logger.warning(
                "One or more dataset splits not found while setting output transforms."
            )

        # build data pipeline
        data_pipeline = DataPipeline(dataset=dataset)

        return ModelExplainerState(
            data_pipeline=data_pipeline,
            x_model_pipeline=x_model_pipeline,
            tb_logger=tb_logger,
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
                logging=self._config.logging,
                test_run=self._config.test_run,
                use_fixed_batch_iterator=self._config.use_fixed_batch_iterator,
                with_amp=self._config.with_amp,
                save_model_outputs_to_disk=self._config.save_test_outputs_to_disk,
            ),
            deps=TestEngineDependencies(
                model_pipeline=self._state.x_model_pipeline._model_pipeline,
                dataloader=test_dataloader,
                device=torch.device(self._device),
                output_dir=self._run_dir,
                tb_logger=self._state.tb_logger,
            ),
        )

    def test(self) -> None:
        output_file_path = self._run_dir / "test_metrics.json"
        if output_file_path.exists():
            logger.info(
                f"Test metrics already exist at {output_file_path}, skipping test."
            )
            return

        # first we generate baseline features on training data if needed
        test_engine = self._build_test_engine()

        # run test engine
        state = test_engine.run(self._checkpoint_path)

        metrics = _format_metrics_for_logging(state.metrics)
        logger.info("Test metrics:")
        logger.info(json.dumps(metrics, indent=4))

        # serialize test metrics
        if not output_file_path.parent.exists():
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file_path, "w") as f:
            json.dump(
                {"config": self._config.model_dump(), "metrics": metrics}, f, indent=4
            )

    def compute_training_baseline_features(self) -> None:
        # first we generate baseline features on training data if needed
        training_baseline_features_generation_engine = (
            self._build_features_generation_engine()
        )

        # generate baseline features
        training_baseline_features_generation_engine.run(self._checkpoint_path)

    def compute_explanations(
        self, total_samples: int | None = None, compute_metrics: bool = False
    ) -> State:
        # then we build the explanation engine first to compute the explanations
        explanation_engine = self._build_explanation_engine(
            total_samples=total_samples, compute_metrics=compute_metrics
        )

        # run explanation engine
        return explanation_engine.run(self._checkpoint_path)

    def run(
        self, total_samples: int | None = None, compute_metrics: bool = False, compute_features_only: bool = False
    ) -> State:
        # run test
        self.test()

        # prepare features
        self.compute_training_baseline_features()

        if compute_features_only:
            logger.info("Feature generation only flag is set. Skipping explanations.")
            return None

        # prepare explanations
        return self.compute_explanations(
            total_samples=total_samples, compute_metrics=compute_metrics
        )
