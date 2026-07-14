from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from atria_logger import get_logger
from atria_models.core.types.model_outputs import TokenClassificationModelOutput
from torch.utils.data import DataLoader

from atria_prv.att._features._extractor import TokenSignalExtractor

if TYPE_CHECKING:
    import torch
    from atria_models.core.model_pipelines._model_pipeline import ModelPipeline

logger = get_logger(__name__)


class SignalExtractionEngine:
    def __init__(
        self,
        model_pipeline: ModelPipeline,
        device: str | torch.device,
        extractor: TokenSignalExtractor,
    ) -> None:
        self._model_pipeline = model_pipeline
        self._device = device
        self._extractor = extractor

    def extract(self, dataloader: DataLoader) -> pd.DataFrame:
        """Return the ``[N, F]`` named feature DataFrame over ``dataloader``."""
        import torch
        from ignite.engine import Engine, Events

        frames: list[pd.DataFrame] = []

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
            output = engine.state.output
            assert isinstance(output, TokenClassificationModelOutput), (
                "This attack is currently only supported for `TokenClassificationModelOutput`"
            )
            frames.append(
                self._extractor.to_features(output.logits, output.token_labels)
            )

        self._model_pipeline.ops.to_device(self._device)
        engine.run(dataloader, max_epochs=1)

        features = pd.concat(frames, ignore_index=True)
        logger.info(
            f"Extracted {features.shape[1]} features for {len(features)} samples."
        )
        return features
