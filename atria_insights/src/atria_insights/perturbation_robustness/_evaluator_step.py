from __future__ import annotations

from collections.abc import Callable

import torch
from atria_logger import get_logger
from atria_ml.training.engine_steps._base import EngineStep
from atria_models import ModelPipeline
from atria_models.core.types.model_outputs import ModelOutput
from atria_transforms.core._data_types._base import TensorDataModel
from ignite.engine import Engine

logger = get_logger(__name__)


class PerturbationRobustnessEvaluatorStep(EngineStep):
    def __init__(
        self,
        model_pipeline: ModelPipeline,
        perturbation_transform: Callable,
        device: str | torch.device,
        test_run: bool = False,
    ):
        super().__init__(
            model_pipeline=model_pipeline,
            device=device,
            with_amp=False,
            test_run=test_run,
        )
        self._perturbation_transform = perturbation_transform

    @property
    def name(self) -> str:
        return "test"

    def __call__(
        self, engine: Engine, batch_list: list[TensorDataModel]
    ) -> ModelOutput:
        """Process batch with optional caching."""
        # set model to eval mode
        with torch.no_grad():
            self._model_pipeline.ops.eval()
            batch = batch_list[0].batch(batch_list)
            batch = batch.ops.to_torch().ops.to(self._device)
            batch = self._perturbation_transform(batch)
            return self._model_pipeline.evaluation_step(
                evaluation_engine=engine, batch=batch, stage="test"
            )
