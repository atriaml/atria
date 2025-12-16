from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from atria_logger import get_logger
from atria_models.core.model_pipelines._model_pipeline import ModelPipeline
from atria_transforms.core._data_types._base import TensorDataModel

if TYPE_CHECKING:
    import torch
    from ignite.engine import Engine


logger = get_logger(__name__)


class EngineStep(ABC):
    def __init__(
        self,
        model_pipeline: ModelPipeline,
        device: str | torch.device,
        with_amp: bool = False,
        test_run: bool = False,
    ):
        super().__init__()

        import torch

        self._model_pipeline = model_pipeline
        self._device = torch.device(device)
        self._with_amp = with_amp
        self._test_run = test_run

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def __call__(
        self, engine: Engine, batch: TensorDataModel
    ) -> Any | tuple[torch.Tensor]:
        """
        Abstract method to execute the engine step.

        Args:
            engine (Engine): The engine executing this step.
            batch (Sequence[torch.Tensor]): The batch of data to process.

        Returns:
            Union[Any, Tuple[torch.Tensor]]: The result of the step execution.
        """
