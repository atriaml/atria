# type: ignore
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from atria_logger import get_logger
from atria_ml.training.engine_steps._base import EngineStep
from atria_transforms.core._data_types._base import TensorDataModel
from ignite.engine import Engine
from pydantic import BaseModel, ConfigDict

from atria_insights.data_types._explanation_inputs import BatchExplanationInputs
from atria_insights.data_types._explanation_state import BatchExplanationState

if TYPE_CHECKING:
    from atria_insights.model_pipelines._model_pipeline import ExplainableModelPipeline


logger = get_logger(__name__)


class ExplanationStepOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    explanation_state: BatchExplanationState
    explanation_inputs: BatchExplanationInputs


class ExplanationStep(EngineStep):
    def __init__(
        self,
        x_model_pipeline: ExplainableModelPipeline,
        device: str | torch.device,
        test_run: bool = False,
    ):
        super().__init__(
            model_pipeline=x_model_pipeline._model_pipeline,
            device=device,
            with_amp=False,
            test_run=test_run,
        )

        self._x_model_pipeline = x_model_pipeline

    @property
    def name(self) -> str:
        return "explanation"

    def __call__(
        self, engine: Engine, batch: list[TensorDataModel]
    ) -> BatchExplanationState:
        """Process batch with optional caching."""
        # set model to eval mode
        self._x_model_pipeline.ops.eval()
        batch = batch[0].batch(batch)
        batch = batch.ops.to(self._device)
        return self._x_model_pipeline.explanation_step(batch=batch)
