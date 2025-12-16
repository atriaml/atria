from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from atria_logger import get_logger
from atria_ml.training.engine_steps._base import EngineStep
from atria_transforms.core._data_types._base import TensorDataModel
from ignite.engine import Engine

from atria_insights.core.model_pipelines._model_pipeline import ExplainableModelPipeline

if TYPE_CHECKING:
    import torch

logger = get_logger(__name__)


class ExplanationStep(EngineStep):
    def __init__(
        self,
        x_model_pipeline: ExplainableModelPipeline,
        device: str | torch.device,
        train_baselines: OrderedDict[str, torch.Tensor] | torch.Tensor | None,
        with_amp: bool = False,
    ):
        self._x_model_pipeline = x_model_pipeline
        self._device = torch.device(device)
        self._train_baselines = train_baselines
        self._with_amp = with_amp

    @property
    def name(self) -> str:
        return "explanation"

    def __call__(
        self, engine: Engine, batch: TensorDataModel
    ) -> Any | tuple[torch.Tensor]:
        import torch
        from torch.cuda.amp.autocast_mode import autocast

        self._x_model_pipeline.ops.eval()
        with torch.no_grad():
            with autocast(enabled=self._with_amp):
                batch = batch.ops.to(self._device)
                return self._x_model_pipeline.explanation_step(
                    batch=batch, train_baselines=self._train_baselines
                )
