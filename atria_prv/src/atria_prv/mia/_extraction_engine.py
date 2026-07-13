from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from atria_logger import get_logger
from atria_models.core.types.model_outputs import TokenClassificationModelOutput
from torch.utils.data import DataLoader

from atria_prv.mia._signals import per_document_signals

if TYPE_CHECKING:
    import torch
    from atria_models.core.model_pipelines._model_pipeline import ModelPipeline

logger = get_logger(__name__)


class SignalExtractionEngine:
    def __init__(
        self, model_pipeline: ModelPipeline, device: str | torch.device
    ) -> None:
        self._model_pipeline = model_pipeline
        self._device = device

    def extract(
        self, dataloader: DataLoader
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import torch
        from ignite.engine import Engine, Events

        probs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        losses: list[np.ndarray] = []

        def step(engine: Engine, batch):
            self._model_pipeline.ops.eval()
            with torch.no_grad():
                collated = batch[0].batch(batch).ops.to_torch().ops.to(self._device)
                return self._model_pipeline.evaluation_step(
                    evaluation_engine=engine, batch=collated, stage="test"
                )

        engine = Engine(step)
        engine.logger.propagate = False

        @engine.on(Events.ITERATION_COMPLETED)
        def _collect(engine: Engine) -> None:
            output = engine.state.output  # TokenClassificationModelOutput
            assert isinstance(output, TokenClassificationModelOutput), (
                "This attack is currently only supported for `TokenClassificationModelOutput`"
            )
            p, y, loss = per_document_signals(
                output.logits, output.token_labels, reduction=self._reduction
            )
            probs.append(p)
            ys.append(y)
            losses.append(loss)

        self._model_pipeline.ops.to_device(self._device)
        engine.run(dataloader, max_epochs=1)
        probs_arr = np.concatenate(probs)
        ys_arr = np.concatenate(ys)
        losses_arr = np.concatenate(losses)
        logger.info(
            f"Extracted signals for {len(probs_arr)} samples "
            f"(feature dim = {probs_arr.shape[1] if probs_arr.ndim > 1 else 1})."
        )
        return probs_arr, ys_arr, losses_arr
