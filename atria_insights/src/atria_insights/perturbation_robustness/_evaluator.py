from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from atria_datasets.core.dataset._datasets import Dataset
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa
from atria_logger._api import get_logger
from atria_ml.data_pipeline._data_pipeline import DataPipeline
from atria_ml.task_pipelines._utilities import (
    _get_env_info,
    _initialize_torch,
    _reset_random_seeds,
)
from atria_ml.training.engines.utilities import _format_metrics_for_logging
from atria_models.core.model_pipelines._model_pipeline import ModelPipeline
from atria_transforms.core._tfs._base import DataTransform
from omegaconf import OmegaConf

from atria_insights.perturbation_robustness._evaluator_engine import (
    PerturbationRobustnessEvaluatorEngine,
    PerturbationRobustnessEvaluatorEngineConfig,
    PerturbationRobustnessEvaluatorEngineDependencies,
)
from atria_insights.perturbation_robustness._task_config import (
    PerturbationRobustnessEvaluatorTaskConfig,
)

if TYPE_CHECKING:
    from ignite.handlers import TensorboardLogger

logger = get_logger(__name__)


@dataclass
class PerturbationRobustnessEvaluatorState:
    data_pipeline: DataPipeline
    model_pipeline: ModelPipeline
    tb_logger: TensorboardLogger | None = None

    @property
    def dataset(self) -> Dataset:
        return self.data_pipeline.dataset


class PerturbationRobustnessEvaluator:
    def __init__(
        self,
        config: PerturbationRobustnessEvaluatorTaskConfig,
        perturbation_transform_generator: Generator[DataTransform, None, None],
        local_rank: int = 0,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self._config = config
        self._checkpoint_path = None
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
            self._run_dir = Path(self._config.env.run_dir) / "default_checkpoint"
        self._state: PerturbationRobustnessEvaluatorState = self._build(
            local_rank=local_rank
        )
        self._perturbation_transform_generator = perturbation_transform_generator

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
            log_dir = Path(self._config.env.run_dir) / "tensorboard"
            log_dir.mkdir(parents=True, exist_ok=True)
            tb_logger = TensorboardLogger(log_dir=log_dir)
        else:
            tb_logger = None
        return tb_logger

    def _build(self, local_rank: int = 0) -> PerturbationRobustnessEvaluatorState:
        self._initialize_runtime(local_rank=local_rank)

        # setup logging
        tb_logger = self._setup_logging()

        # build dataset
        dataset = self._config.data.build_dataset()

        # load labels
        labels = dataset.metadata.dataset_labels

        # log dataset info
        logger.info(f"Dataset:\n{dataset}")

        # build model pipeline
        model_pipeline = self._config.model_pipeline.build(labels=labels)

        # log model pipeline
        logger.info(model_pipeline.ops.summarize())

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

        return PerturbationRobustnessEvaluatorState(
            data_pipeline=data_pipeline,
            model_pipeline=model_pipeline,
            tb_logger=tb_logger,
        )

    def _build_test_engine(
        self, perturbation_transform: Callable
    ) -> PerturbationRobustnessEvaluatorEngine:
        import torch

        test_dataloader = self._state.data_pipeline.test_dataloader(
            batch_size=self._config.data.eval_batch_size,
            num_workers=self._config.data.num_workers,
            pin_memory=self._config.data.pin_memory,
        )
        return PerturbationRobustnessEvaluatorEngine(
            config=PerturbationRobustnessEvaluatorEngineConfig(
                logging=self._config.logging,
                test_run=self._config.test_run,
                use_fixed_batch_iterator=self._config.use_fixed_batch_iterator,
                with_amp=self._config.with_amp,
            ),
            deps=PerturbationRobustnessEvaluatorEngineDependencies(
                model_pipeline=self._state.model_pipeline,
                dataloader=test_dataloader,
                device=torch.device(self._device),
                output_dir=self._run_dir,
                tb_logger=self._state.tb_logger,
                perturbation_transform=perturbation_transform,
            ),
        )

    def run(self):
        # intialize output file
        output_file_path = self._run_dir / "metrics.json"
        if not output_file_path.exists():
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file_path, "w") as f:
                json.dump({}, f)

        for perturbation_transform in self._perturbation_transform_generator:
            for run_idx in range(self._config.n_runs_per_perturbation):
                _reset_random_seeds(run_idx)

                logger.debug(
                    f"Starting run {run_idx + 1}/{self._config.n_runs_per_perturbation} for baseline generator "
                    f"{perturbation_transform}."
                )

                # build the transform if it's a partial
                perturbation_transform.build(model=self._state.model_pipeline._model)

                # first we generate baseline features on training data if needed
                test_engine = self._build_test_engine(
                    perturbation_transform=perturbation_transform
                )

                # run test engine
                state = test_engine.run(self._checkpoint_path)

                # get formatted metrics
                metrics = _format_metrics_for_logging(state.metrics)

                # get config hash for this run
                config_hash = hashlib.sha256(
                    json.dumps(
                        {
                            "baseline_generator": perturbation_transform.model_dump(),
                            "run_idx": run_idx,
                        },
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest()[:8]

                # append metrics to output file
                with open(output_file_path, "r+") as f:
                    all_metrics = json.load(f)
                    all_metrics[config_hash] = {
                        "perturbation_transform": perturbation_transform.model_dump(),
                        "run_idx": run_idx,
                        "metrics": metrics,
                    }
                    f.seek(0)
                    json.dump(all_metrics, f, indent=4)
