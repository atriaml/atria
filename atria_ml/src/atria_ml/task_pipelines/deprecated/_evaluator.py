"""
This module defines the `ModelEvaluator` class, which is responsible for evaluating a machine learning model
using a specified data pipeline, task module, and test engine. The class is designed to be registered as a
task runner in the `atria` framework and provides functionality for initializing and running the evaluation process.

Classes:
    - ModelEvaluator: A task runner for evaluating models.

Constants:
    - MODEL_EVALUATOR_OUTPUT_DIR: A string template for the output directory structure.

Dependencies:
    - ignite.distributed: For distributed training and device management.
    - atria_datasets.data_pipelines.default_pipeline.DefaultDataPipeline: For handling data pipelines.
    - atria_ml.logger.logger.get_logger: For logging.
    - atria_models.model_pipelines.atria_model_pipeline.AtriaTaskModule: For task-specific model handling.
    - atria_ml.training.engines.evaluation.TestEngine: For running the evaluation process.
    - atria_ml.training.utilities.constants.TrainingStage: For defining training stages.
    - atria_ml.training.utilities.initialization: For initializing PyTorch and TensorBoard.
    - atria_ml.utilities.common._msg_with_separator: For formatting log messages.

Usage:
    The `ModelEvaluator` class is registered as a task runner and can be instantiated with the required
    components (data pipeline, task module, and test engine). It provides a `run` method to execute the
    evaluation process and an `_initialize` method for setting up the necessary components.
"""

from pathlib import Path

from atria_core.logger.logger import get_logger
from atria_core.types import DatasetSplitType
from atria_datasets.pipelines.atria_data_pipeline import AtriaDataPipeline
from atria_models.pipelines.atria_model_pipeline import AtriaModelPipeline

from atria_ml.registry import TASK_PIPELINE
from atria_ml.training.engines.evaluation import TestEngine

logger = get_logger(__name__)


@TASK_PIPELINE.register(
    "evaluator",
    defaults=[
        "_self_",
        {"/data_pipeline@data_pipeline": "default"},
        {"/model_pipeline@model_pipeline": "image_classification"},
        {"/engine@test_engine": "default_test_engine"},
    ],
    zen_meta={
        "hydra": {
            "run": {"dir": "${checkpoint_dir}"},
            "output_subdir": "hydra",
            "job": {"chdir": False},
            "searchpath": ["pkg://atria/conf", "pkg://atria_examples/conf"],
        }
    },
    is_global_package=True,
)
class Evaluator:
    """
    A task runner for evaluating machine learning models.

    Args:
        run_dir (str): The directory where the model checkpoints are stored.
        data_pipeline (DefaultDataPipeline): The data pipeline for loading and processing data.
        model_pipeline (partial[AtriaTaskModule]): A partially initialized task module for handling the model.
        test_engine (partial[TestEngine]): A partially initialized test engine for running the evaluation.
        run_dir (str): The directory where evaluation outputs will be saved and where checkpoints are stored.
        seed (int): The random seed for reproducibility.
        deterministic (bool): Whether to use deterministic algorithms.
        backend (Optional[str]): The backend for distributed training (e.g., "nccl").
        n_devices (int): The number of devices to use for evaluation.

    Attributes:
        run_dir (str): The directory where the model checkpoints are stored.
        data_pipeline (DefaultDataPipeline): The data pipeline for loading and processing data.
        model_pipeline (partial[AtriaTaskModule]): A partially initialized task module for handling the model.
        test_engine (partial[TestEngine]): A partially initialized test engine for running the evaluation.
        run_dir (str): The directory where evaluation outputs will be saved.
        seed (int): The random seed for reproducibility.
        deterministic (bool): Whether to use deterministic algorithms.
        backend (Optional[str]): The backend for distributed training (e.g., "nccl").
        n_devices (int): The number of devices to use for evaluation.
    """

    def __init__(
        self,
        data_pipeline: AtriaDataPipeline,
        model_pipeline: AtriaModelPipeline,
        test_engine: TestEngine,
        run_dir: str,
        seed: int = 42,
        deterministic: bool = False,
        backend: str | None = "nccl",
        n_devices: int = 1,
    ):
        self._data_pipeline = data_pipeline
        self._run_dir = Path(run_dir)
        self._model_pipeline = model_pipeline
        self._test_engine = test_engine
        self._seed = seed
        self._deterministic = deterministic
        self._backend = backend
        self._n_devices = n_devices

    # ---------------------------
    # Public Methods
    # ---------------------------

    def build(self) -> None:
        """
        Initializes the components required for evaluation, including logging, data pipeline, task module,
        and test engine.
        """
        self._initialize_runtime()
        self._setup_logging()
        self._build_data_pipeline()
        self._build_model_pipeline()
        self._build_test_engine()

    def run(self) -> None:
        """
        Executes the evaluation process by running the test engine.
        """
        self._test_engine.run()

    # ---------------------------
    # Private Methods
    # ---------------------------

    def _initialize_runtime(self) -> None:
        import ignite.distributed as idist

        from atria_ml.training.utilities.torch_utils import _initialize_torch

        # initialize training
        _initialize_torch(seed=self._seed, deterministic=self._deterministic)

        # initialize torch device (cpu or gpu)
        self._device = idist.device()

    def _setup_logging(self) -> None:
        import logging

        from atria_ml.training.utilities.torch_utils import _setup_tensorboard

        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                logger.info(
                    f"Verbose logs can be found at file: {handler.baseFilename}"
                )

        # initialize logging directory and tensorboard logger
        self._tb_logger = _setup_tensorboard(self._run_dir)

    def _build_data_pipeline(self) -> AtriaDataPipeline:
        """
        Builds the data pipeline for the evaluation process.
        """

        from atria_core.utilities.common import _msg_with_separator

        # Set up the data pipeline
        logger.info(_msg_with_separator("Setting up data pipeline"))
        self._data_pipeline = self._data_pipeline.build(split=DatasetSplitType.test)

    def _build_model_pipeline(self) -> AtriaModelPipeline:
        """
        Builds the model pipeline for the evaluation process.
        """

        # Set up the model pipeline
        self._model_pipeline = self._model_pipeline.build(
            dataset_metadata=self._data_pipeline.dataset_metadata,
            tb_logger=self._tb_logger,
        )

    def _build_test_engine(self) -> TestEngine:
        """
        Builds the test engine for the evaluation process.
        """

        # Initialize the test engine from the partial
        logger.info("Setting up test engine")
        self._test_engine = self._test_engine.build(
            output_dir=self._run_dir,
            model_pipeline=self._model_pipeline,
            dataloader=self._data_pipeline.test_dataloader(),
            device=self._device,
            tb_logger=self._tb_logger,
        )
