# type: ignore
from __future__ import annotations

import torch
from atria_logger import get_logger
from atria_ml.training.engine_steps._base import EngineStep
from atria_models.core.types.model_outputs import ModelOutput
from atria_transforms.core._data_types._base import TensorDataModel
from ignite.engine import Engine

from atria_insights.feature_perturbation.pipelines._base import (
    FeaturePerturbationPipeline,
)

logger = get_logger(__name__)


class FeaturePerturbationEvaluatorStep(EngineStep):
    def __init__(
        self,
        fp_pipeline: FeaturePerturbationPipeline,
        device: str | torch.device,
        test_run: bool = False,
    ):
        super().__init__(
            model_pipeline=fp_pipeline._model_pipeline,
            device=device,
            with_amp=False,
            test_run=test_run,
        )

        self._fp_pipeline = fp_pipeline

    def __call__(self, engine: Engine, batch: list[TensorDataModel]) -> ModelOutput:
        """Process batch with optional caching."""
        # set model to eval mode
        self._model_pipeline.ops.eval()
        batch = batch[0].batch(batch)
        batch = batch.ops.to(self._device)
        return self._fp_pipeline.perturb_and_evaluate(batch=batch)
