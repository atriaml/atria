"""
This module defines the base class for engine steps used in the training process.

The `BaseEngineStep` class is an abstract base class that provides a blueprint for implementing specific steps in the training engine.
It includes methods for attaching a parent engine, building the step with a model pipeline and device, and an abstract method for executing the step.

Classes:
    BaseEngineStep: Abstract base class for defining engine steps.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from atria_core.logger.logger import get_logger

from atria_ml.training.utilities.constants import TrainingStage

if TYPE_CHECKING:
    import torch
    from ignite.engine import Engine


logger = get_logger(__name__)


class BaseEngineStep(ABC):
    """
    Abstract base class for defining engine steps in the training process.

    Attributes:
        _parent_engine (Engine): The parent engine to which this step is attached.
    """

    def __init__(self):
        """
        Initializes the BaseEngineStep instance.
        """
        self._parent_engine = None

    def attach_parent_engine(self, engine: "Engine") -> None:
        """
        Attaches a parent engine to this step.

        Args:
            engine (Engine): The parent engine to attach.
        """
        self._parent_engine = engine

    @property
    @abstractmethod
    def stage(self) -> TrainingStage:
        """
        Abstract property that defines the training stage for this engine step.

        Returns:
            TrainingStage: The training stage associated with this step.
        """

    @abstractmethod
    def __call__(
        self, engine: "Engine", batch: Sequence["torch.Tensor"]
    ) -> Any | tuple["torch.Tensor"]:
        """
        Abstract method to execute the engine step.

        Args:
            engine (Engine): The engine executing this step.
            batch (Sequence[torch.Tensor]): The batch of data to process.

        Returns:
            Union[Any, Tuple[torch.Tensor]]: The result of the step execution.
        """
