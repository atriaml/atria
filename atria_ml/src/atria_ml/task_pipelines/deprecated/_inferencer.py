"""
This module defines the `ModelInferencer` class, which is responsible for evaluating a machine learning model
using a specified data pipeline, task module, and test engine. The class is designed to be registered as a
task runner in the `atria` framework and provides functionality for initializing and running the evaluation process.

Classes:
    - ModelInferencer: A task runner for evaluating models.

Dependencies:
    - ignite.distributed: For distributed training and device management.
    - atria_datasets.data_pipelines.default_pipeline.DefaultDataPipeline: For handling data pipelines.
    - atria_ml.logger.logger.get_logger: For logging.
    - atria_models.model_pipelines.atria_model_pipeline.AtriaTaskModule: For task-specific model handling.
    - atria_ml.registry.TASK_RUNNER: For registering the task runner.
    - atria_ml.training.engines.evaluation.InferenceEngine: For running the evaluation process.
    - atria_ml.training.utilities.constants.TrainingStage: For defining training stages.
    - atria_ml.training.utilities.initialization: For initializing PyTorch and TensorBoard.
    - atria_ml.utilities.common._msg_with_separator: For formatting log messages.

Usage:
    The `ModelInferencer` class is registered as a task runner and can be instantiated with the required
    components (data pipeline, task module, and test engine). It provides a `run` method to execute the
    evaluation process and an `_initialize` method for setting up the necessary components.
"""

from collections.abc import Iterator
from functools import partial
from typing import ClassVar

from atria_core.logger.logger import get_logger
from atria_core.types import TaskType
from atria_core.utilities.repr import RepresentationMixin
from atria_datasets.pipelines.utilities import (
    auto_dataloader,
    default_collate,
    mmdet_pseudo_collate,
)
from atria_models.pipelines.atria_model_pipeline import AtriaModelPipeline

from atria_ml.registry import TASK_PIPELINE
from atria_ml.training.engines.evaluation import InferenceEngine

logger = get_logger(__name__)


@TASK_PIPELINE.register(
    "inferencer",
    zen_meta={
        "hydra": {
            "run": {
                "dir": "outputs/inferencer/${dataset.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}"
            },
            "output_subdir": "hydra",
            "job": {"chdir": False},
            "searchpath": ["pkg://atria/conf", "pkg://atria_examples/conf"],
        },
        "n_devices": 1,
        "backend": "nccl",
    },
    is_global_package=True,
)
class Inferencer(RepresentationMixin):
    _REGISTRY_CONFIGS: ClassVar[type[dict]] = {
        "image_classification": {
            "hydra_defaults": [
                {"/engine@inference_engine": "default_inference_engine"},
                "_self_",
            ]
        },
        "sequence_classification": {
            "hydra_defaults": [
                {"/engine@inference_engine": "default_inference_engine"},
                "_self_",
            ]
        },
        "token_classification": {
            "hydra_defaults": [
                {"/engine@inference_engine": "default_inference_engine"},
                "_self_",
            ]
        },
        "layout_token_classification": {
            "hydra_defaults": [
                {"/engine@inference_engine": "default_inference_engine"},
                "_self_",
            ]
        },
        "visual_question_answering": {
            "hydra_defaults": [
                {"/engine@inference_engine": "default_inference_engine"},
                "_self_",
            ]
        },
        "layout_analysis": {
            "hydra_defaults": [
                {"/engine@inference_engine": "default_inference_engine"},
                "_self_",
            ],
            "collate_fn": "mmdet_pseudo_collate",
        },
    }
    """
    A task runner for evaluating machine learning models.

    Args:
        inference_engine (partial[InferenceEngine]): A partially initialized test engine for running the evaluation.
        evaluation_dataloader (partial): A partial function for creating the evaluation data loader.
        allowed_keys (set): A set of allowed keys for the data pipeline.
        collate_fn (str): The collate function to use for data loading.


    Attributes:
        _inference_engine (partial[InferenceEngine]): A partially initialized test engine for running the evaluation.
        _runtime_transforms (DataTransformsDict): A dictionary of data transforms for runtime evaluation.
        _evaluation_dataloader (partial): A partial function for creating the evaluation data loader.
        _allowed_keys (set): A set of allowed keys for the data pipeline.
        _collate_fn (str): The collate function to use for data loading.
    """

    def __init__(
        self,
        inference_engine: InferenceEngine,
        evaluation_dataloader: partial = partial(
            auto_dataloader,
            batch_size=64,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        ),
        allowed_keys: set | None = None,
        collate_fn: str = "default_collate",
    ):
        self._inference_engine = inference_engine
        self._evaluation_dataloader = evaluation_dataloader
        self._allowed_keys = allowed_keys

        assert collate_fn in ["default_collate", "mmdet_pseudo_collate"], (
            f"collate_fn must be one of ['collate_fn', 'mmdet_pseudo_collate'], "
            f"but got {collate_fn}"
        )
        if collate_fn == "default_collate":
            self._collate_fn = default_collate
        elif collate_fn == "mmdet_pseudo_collate":
            self._collate_fn = mmdet_pseudo_collate

        self._model_pipeline: AtriaModelPipeline = None
        self._compute_metrics: bool = False

    # ---------------------------
    # Public Methods
    # ---------------------------

    @classmethod
    def load_from_registry(
        cls,
        task_type: TaskType,
        model_pipeline: AtriaModelPipeline,
        compute_metrics: bool = False,
    ):
        """
        Loads the inferencer from the registry based on the module name or task type.
        """
        inferencer: Inferencer = TASK_PIPELINE.load_from_registry(
            module_name=f"inferencer/{task_type.value}"
        )
        assert inferencer._task_type == model_pipeline.task_type, (
            f"Task type mismatch: {inferencer._task_type} != {model_pipeline.task_type}"
        )
        return inferencer.build(
            model_pipeline=model_pipeline, compute_metrics=compute_metrics
        )

    def build(
        self, model_pipeline: AtriaModelPipeline, compute_metrics: bool = False
    ) -> None:
        """
        Initializes the components required for evaluation, including logging, data pipeline, task module,
        and test engine.
        """
        self._initialize()
        self._build_inference_engine(
            model_pipeline=model_pipeline, compute_metrics=compute_metrics
        )
        return self

    def run(self, dataset: Iterator) -> None:
        data_loader = self._evaluation_dataloader(
            dataset, num_workers=0, collate_fn=self._collate_fn
        )
        return self._inference_engine.run(data_loader)

    def _initialize(self):
        import ignite.distributed as idist

        from atria_ml.training.utilities.torch_utils import _initialize_torch

        _initialize_torch()
        self._device = idist.device()

    def _build_inference_engine(
        self, model_pipeline: AtriaModelPipeline, compute_metrics: bool = False
    ) -> InferenceEngine:
        """
        Builds the test engine for the evaluation process.
        """

        # Initialize the test engine from the partial
        logger.info("Setting up inference engine")
        self._inference_engine = self._inference_engine.build(
            model_pipeline=model_pipeline, device=self._device
        )
