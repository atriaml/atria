from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from atria_core.logger import get_logger
from atria_core.types import TaskType
from atria_datasets.pipelines.atria_data_pipeline import AtriaDataPipeline
from atria_models.pipelines.atria_model_pipeline import AtriaModelPipeline
from atria_registry.registry_config import RegistryConfig

from atria_ml.registry import TASK_PIPELINE
from atria_ml.training.engines.atria_engine import AtriaEngine
from atria_ml.training.engines.evaluation import TestEngine
from atria_ml.training.engines.utilities import RunConfig

if TYPE_CHECKING:
    from ignite.engine import State

logger = get_logger(__name__)

DEFAULTS_SELF = ["_self_"]
DATA_PIPELINE_DEFAULT_LIST = [{"/data_pipeline@data_pipeline": "default"}]
MODEL_DEFAULT_LIST = [{"/model_pipeline@model_pipeline": "???"}]
ENGINE_DEFAULT_LIST = [{"/engine@test_engine": "default_test_engine"}]


@TASK_PIPELINE.register(
    "evaluator",
    configs=[
        RegistryConfig(
            name="default",
            defaults=DEFAULTS_SELF
            + ENGINE_DEFAULT_LIST
            + DATA_PIPELINE_DEFAULT_LIST
            + MODEL_DEFAULT_LIST,
            output_dir="outputs/evaluator/${resolve_experiment_name:${experiment_name}}",
        ),
        RegistryConfig(
            name=TaskType.image_classification.value,
            defaults=DEFAULTS_SELF
            + ENGINE_DEFAULT_LIST
            + DATA_PIPELINE_DEFAULT_LIST
            + [{"/model_pipeline@model_pipeline": "image_classification"}],
            output_dir="outputs/evaluator/image_classification/${resolve_experiment_name:${experiment_name}}/",
        ),
        RegistryConfig(
            name=TaskType.sequence_classification.value,
            defaults=DEFAULTS_SELF
            + ENGINE_DEFAULT_LIST
            + DATA_PIPELINE_DEFAULT_LIST
            + [{"/model_pipeline@model_pipeline": "sequence_classification"}],
            output_dir="outputs/evaluator/sequence_classification/${resolve_experiment_name:${experiment_name}}/",
        ),
        RegistryConfig(
            name=TaskType.token_classification.value,
            defaults=DEFAULTS_SELF
            + ENGINE_DEFAULT_LIST
            + DATA_PIPELINE_DEFAULT_LIST
            + [{"/model_pipeline@model_pipeline": "token_classification"}],
            output_dir="outputs/evaluator/token_classification/${resolve_experiment_name:${experiment_name}}/",
        ),
        RegistryConfig(
            name=TaskType.layout_token_classification.value,
            defaults=DEFAULTS_SELF
            + ENGINE_DEFAULT_LIST
            + DATA_PIPELINE_DEFAULT_LIST
            + [{"/model_pipeline@model_pipeline": "layout_token_classification"}],
            output_dir="outputs/evaluator/layout_token_classification/${resolve_experiment_name:${experiment_name}}/",
        ),
        RegistryConfig(
            name=TaskType.visual_question_answering.value,
            defaults=DEFAULTS_SELF
            + ENGINE_DEFAULT_LIST
            + DATA_PIPELINE_DEFAULT_LIST
            + [{"/model_pipeline@model_pipeline": "visual_question_answering"}],
            output_dir="outputs/evaluator/visual_question_answering/${resolve_experiment_name:${experiment_name}}/",
        ),
        RegistryConfig(
            name=TaskType.layout_analysis.value,
            defaults=DEFAULTS_SELF
            + DATA_PIPELINE_DEFAULT_LIST
            + ENGINE_DEFAULT_LIST
            + [{"/model_pipeline@model_pipeline": "object_detection"}],
            output_dir="outputs/evaluator/layout_analysis/${resolve_experiment_name:${experiment_name}}/",
            data_pipeline={"collate_fn": "mmdet_pseudo_collate"},
        ),
    ],
    zen_meta={
        "pipeline_name": "atria_evaluator",
        "n_devices": 1,
        "backend": "nccl",
        "experiment_name": "_to_be_resolved_",
    },
    zen_exclude=["hydra", "package", "version"],
    is_global_package=True,
)
class Evaluator:
    def __init__(
        self,
        test_checkpoint: str,
        data_pipeline: AtriaDataPipeline,
        model_pipeline: AtriaModelPipeline,
        test_engine: TestEngine,
        output_dir: str | None = None,
        prepare_atria_ckpt: bool = True,
        do_eval: bool = True,
    ):
        self._output_dir = output_dir
        self._data_pipeline = data_pipeline
        self._model_pipeline = model_pipeline
        self._test_engine = test_engine
        self._test_checkpoint = test_checkpoint
        self._prepare_atria_ckpt = prepare_atria_ckpt
        self._do_eval = do_eval

    @property
    def model_pipeline(self):
        return self._model_pipeline

    @property
    def test_engine(self):
        return self._test_engine

    @property
    def device(self):
        return self._device

    @property
    def logger(self):
        return logger

    def _initialize_runtime(self, local_rank: int) -> None:
        self._device = local_rank

    def _setup_logging(self) -> None:
        from atria_ml.training.utilities.torch_utils import _setup_tensorboard

        # initialize logging directory and tensorboard logger
        if self._output_dir is not None:
            self._tb_logger = _setup_tensorboard(self._output_dir)

    def _build_data_pipeline(self):
        logger.info("Setting up data pipeline")
        self._data_pipeline.build(
            runtime_transforms=self._model_pipeline.config.runtime_transforms
        )

    def _build_model_pipeline(self) -> None:
        # initialize the task module from partial
        logger.info("Setting up task module")
        self._model_pipeline = self._model_pipeline.build(
            dataset_metadata=self._data_pipeline.dataset_metadata,
            tb_logger=self._tb_logger,
        )
        logger.info(self._model_pipeline)

    def _build_test_engine(self) -> None:
        if self._test_engine is not None:
            logger.info("Setting up test engine")
            logger.info(
                "Initializing train dataloader with config: %s",
                self._data_pipeline._dataloader_config.eval_config,
            )
            test_dataloader = self._data_pipeline.test_dataloader()
            self._test_engine: AtriaEngine = self._test_engine.build(
                output_dir=self._output_dir,
                model_pipeline=self._model_pipeline,
                dataloader=test_dataloader,
                device=self._device,
                tb_logger=self._tb_logger,
            )

    def build(
        self, local_rank: int, experiment_name: str, run_config: RunConfig
    ) -> None:
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                logger.info(f"Log file path: {handler.baseFilename}")

        self._experiment_name = experiment_name
        self._run_config = run_config
        self._initialize_runtime(local_rank=local_rank)
        self._setup_logging()
        self._build_data_pipeline()
        self._build_model_pipeline()

    def test(self) -> dict[str, State]:
        self._build_test_engine()
        return self._test_engine.run(test_checkpoint=self._test_checkpoint)

    def _save_checkpoint(self, state: dict[str, State] | None = None) -> None:
        from pathlib import Path

        import torch

        from atria_ml.training.engines.utilities import (
            EXPERIMENT_NAME_KEY,
            METRICS_KEY,
            MODEL_PIPELINE_CHECKPOINT_KEY,
            RUN_CONFIG_KEY,
        )

        checkpoint = {
            EXPERIMENT_NAME_KEY: self._experiment_name,
            RUN_CONFIG_KEY: self._run_config.state_dict(),
            MODEL_PIPELINE_CHECKPOINT_KEY: self._model_pipeline.state_dict(),
        }
        if state is not None:
            checkpoint[METRICS_KEY] = {
                k: v.metrics for k, v in state.items() if v is not None
            }
        checkpoint_name = Path(self._test_checkpoint).with_suffix("").name
        if self._output_dir is not None:
            checkpoint_path = Path(self._output_dir) / (
                "atria-" + checkpoint_name + ".ckpt"
            )
        else:
            checkpoint_path = Path(self._test_checkpoint) / (
                "atria-" + checkpoint_name + ".ckpt"
            )
        logger.info(
            f"Saving evaluation checkpoint to: {checkpoint_path} with following keys: {checkpoint.keys()}"
        )
        torch.save(checkpoint, checkpoint_path)

    def run(self) -> dict[str, State]:
        assert self._prepare_atria_ckpt or self._do_eval, (
            "Either `prepare_atria_ckpt` or `do_eval` must be True. "
        )
        state = self.test() if self._do_eval else None
        if self._prepare_atria_ckpt:
            self._save_checkpoint(state=state)
        return state
