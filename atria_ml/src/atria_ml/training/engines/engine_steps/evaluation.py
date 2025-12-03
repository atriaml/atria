"""
This module defines various engine steps for evaluation, validation, visualization, testing, inference, and feature extraction.

The `EvaluationStep` class serves as a base class for implementing specific evaluation-related steps. Subclasses such as `ValidationStep`, `VisualizationStep`, `TestStep`, `InferenceStep`, and `FeatureExtractorStep` provide concrete implementations for different stages of the training and evaluation process.

Classes:
    EvaluationStep: Base class for evaluation-related engine steps.
    ValidationStep: Engine step for validation.
    VisualizationStep: Engine step for visualization.
    TestStep: Engine step for testing.
    InferenceStep: Engine step for inference.
    FeatureExtractorStep: Engine step for feature extraction.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Union

from atria_core.logger.logger import get_logger

from atria_ml.training.engines.engine_steps.base import BaseEngineStep
from atria_ml.training.utilities.constants import TrainingStage

if TYPE_CHECKING:
    import torch
    from atria_core.types import BaseDataInstance
    from atria_models.pipelines.atria_model_pipeline import AtriaModelPipeline
    from ignite.engine import Engine


logger = get_logger(__name__)


class EvaluationStep(BaseEngineStep):
    """
    Base class for evaluation-related engine steps.

    This class provides a common interface for executing evaluation steps, including handling model evaluation mode, AMP (automatic mixed precision), and device placement.
    Attributes:
        _model_pipeline (AtriaModelPipeline): The model pipeline to use.
        _device (Union[str, torch.device]): The device to use.
        _with_amp (bool): Whether AMP is enabled. Defaults to False.
        _test_run (bool): Whether this is a test run. Defaults to False.
    """

    def __init__(
        self,
        model_pipeline: "AtriaModelPipeline",
        device: Union[str, "torch.device"],
        with_amp: bool = False,
        test_run: bool = False,
    ):
        """
        Initializes the TrainingStep instance.

        Args:
            model_pipeline (AtriaModelPipeline): The model pipeline to use.
            device (Union[str, torch.device]): The device to use.
            with_amp (bool): Whether AMP is enabled. Defaults to False.
            test_run (bool): Whether this is a test run. Defaults to False.
        """
        super().__init__()

        import torch

        self._model_pipeline = model_pipeline
        self._device = torch.device(device)
        self._with_amp = with_amp
        self._test_run = test_run

    def __call__(
        self, engine: "Engine", batch: "BaseDataInstance", **kwargs
    ) -> Any | tuple["torch.Tensor"]:
        """
        Executes the evaluation step.

        Args:
            engine (Engine): The engine executing this step.
            batch (BaseDataInstance): The batch of data to process.
            **kwargs: Additional keyword arguments for the step.

        Returns:
            Union[Any, Tuple[torch.Tensor]]: The result of the evaluation step.
        """
        import torch
        from torch.cuda.amp import autocast

        self._model_pipeline.eval()
        if self._with_amp:
            self._model_pipeline.half()

        with torch.no_grad():
            with autocast(enabled=self._with_amp):
                if hasattr(batch, "to_device"):
                    batch = batch.to_device(self._device)
                return self._model_step(engine=engine, batch=batch, **kwargs)

    def _model_step(
        self,
        engine: "Engine",
        batch: Sequence["torch.Tensor"],
        test_run: bool = False,
        **kwargs,
    ) -> Any | tuple["torch.Tensor"]:
        """
        Abstract method to perform the model-specific step.

        Args:
            engine (Engine): The engine executing this step.
            batch (Sequence[torch.Tensor]): The batch of data to process.
            test_run (bool): Whether this is a test run. Defaults to False.
            **kwargs: Additional keyword arguments for the step.

        Returns:
            Union[Any, Tuple[torch.Tensor]]: The result of the model step.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method")


class ValidationStep(EvaluationStep):
    """
    Engine step for validation.

    This step is responsible for executing the validation logic during the training process.
    """

    @property
    def stage(self) -> TrainingStage:
        """
        Returns the training stage for this step.

        Returns:
            TrainingStage: The validation stage.
        """
        return TrainingStage.validation

    def _model_step(
        self, engine: "Engine", batch: Sequence["torch.Tensor"], test_run: bool = False
    ) -> Any | tuple["torch.Tensor"]:
        """
        Performs the validation step.

        Args:
            engine (Engine): The engine executing this step.
            batch (Sequence[torch.Tensor]): The batch of data to process.
            test_run (bool): Whether this is a test run. Defaults to False.

        Returns:
            Union[Any, Tuple[torch.Tensor]]: The result of the validation step.
        """
        # forward pass
        return self._model_pipeline.evaluation_step(
            evaluation_engine=engine,
            training_engine=self._parent_engine,
            batch=batch,
            stage=self.stage,
            test_run=test_run,
        )


class VisualizationStep(EvaluationStep):
    """
    Engine step for visualization.

    This step is responsible for executing the visualization logic during the training process.
    """

    @property
    def stage(self) -> TrainingStage:
        """
        Returns the training stage for this step.

        Returns:
            TrainingStage: The visualization stage.
        """
        return TrainingStage.visualization

    def _model_step(
        self, engine: "Engine", batch: Sequence["torch.Tensor"], test_run: bool = False
    ) -> Any | tuple["torch.Tensor"]:
        """
        Performs the visualization step.

        Args:
            engine (Engine): The engine executing this step.
            batch (Sequence[torch.Tensor]): The batch of data to process.
            test_run (bool): Whether this is a test run. Defaults to False.

        Returns:
            Union[Any, Tuple[torch.Tensor]]: The result of the visualization step.
        """
        # forward pass
        return self._model_pipeline.visualization_step(
            evaluation_engine=engine,
            training_engine=self._parent_engine,
            batch=batch,
            stage=self.stage,
            test_run=test_run,
        )


class TestStep(EvaluationStep):
    """
    Engine step for testing.

    This step is responsible for executing the testing logic during the training process.
    """

    @property
    def stage(self) -> TrainingStage:
        """
        Returns the training stage for this step.

        Returns:
            TrainingStage: The testing stage.
        """
        return TrainingStage.test

    def _model_step(
        self, engine: "Engine", batch: Sequence["torch.Tensor"], test_run: bool = False
    ) -> Any | tuple["torch.Tensor"]:
        """
        Performs the testing step.

        Args:
            engine (Engine): The engine executing this step.
            batch (Sequence[torch.Tensor]): The batch of data to process.
            test_run (bool): Whether this is a test run. Defaults to False.

        Returns:
            Union[Any, Tuple[torch.Tensor]]: The result of the testing step.
        """
        # forward pass
        return self._model_pipeline.evaluation_step(
            evaluation_engine=engine, batch=batch, stage=self.stage, test_run=test_run
        )


class InferenceStep(EvaluationStep):
    """
    Engine step for inference.

    This step is responsible for executing the inference logic during the training process.
    """

    @property
    def stage(self) -> TrainingStage:
        """
        Returns the training stage for this step.

        Returns:
            TrainingStage: The inference stage.
        """
        return TrainingStage.predict

    def _model_step(
        self, engine: "Engine", batch: Sequence["torch.Tensor"], test_run: bool = False
    ) -> Any | tuple["torch.Tensor"]:
        """
        Performs the inference step.

        Args:
            engine (Engine): The engine executing this step.
            batch (Sequence[torch.Tensor]): The batch of data to process.
            test_run (bool): Whether this is a test run. Defaults to False.

        Returns:
            Union[Any, Tuple[torch.Tensor]]: The result of the inference step.
        """
        # forward pass
        # inferencer can also be used to evaluate the metrics
        return self._model_pipeline.evaluation_step(
            evaluation_engine=engine, batch=batch, stage=self.stage, test_run=test_run
        )


class FeatureExtractorStep(EvaluationStep):
    """
    Engine step for feature extraction.

    This step is responsible for executing the feature extraction logic during the training process.
    """

    @property
    def stage(self) -> str:
        """
        Returns the training stage for this step.

        Returns:
            str: The feature extraction stage.
        """
        return "FeatureExtractor"

    def _model_step(
        self, engine: "Engine", batch: Sequence["torch.Tensor"], test_run: bool = False
    ) -> Any | tuple["torch.Tensor"]:
        """
        Performs the feature extraction step.

        Args:
            engine (Engine): The engine executing this step.
            batch (Sequence[torch.Tensor]): The batch of data to process.
            test_run (bool): Whether this is a test run. Defaults to False.

        Returns:
            Union[Any, Tuple[torch.Tensor]]: The result of the feature extraction step.

        Raises:
            AssertionError: If the model pipeline does not have a `feature_extractor_step` method.
        """
        assert hasattr(self._model_pipeline, "feature_extractor_step"), (
            f"Task module [{self._model_pipeline.__class__.__name__}] "
            f"does not have method [feature_extractor_step]."
        )

        return self._model_pipeline.feature_extractor_step(
            engine=engine, batch=batch, stage=self.stage, test_run=test_run
        )
