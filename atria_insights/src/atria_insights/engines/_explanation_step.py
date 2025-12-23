from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from atria_logger import get_logger
from atria_ml.training.engine_steps._base import EngineStep
from atria_transforms.core._data_types._base import TensorDataModel
from ignite.engine import Engine

from atria_insights.data_types._explanation_sate import (
    BatchExplanationState,
    SampleExplanationState,
)
from atria_insights.storage.sample_cache_managers._explanation_state import (
    ExplanationStateCacher,
)

if TYPE_CHECKING:
    from atria_insights.model_pipelines._model_pipeline import ExplainableModelPipeline


logger = get_logger(__name__)


class ExplanationStep(EngineStep):
    def __init__(
        self,
        x_model_pipeline: ExplainableModelPipeline,
        device: str | torch.device,
        enable_outputs_caching: bool = False,
        cache_dir: str | Path | None = None,
        test_run: bool = False,
    ):
        super().__init__(
            model_pipeline=x_model_pipeline._model_pipeline,
            device=device,
            with_amp=False,
            test_run=test_run,
        )

        self._x_model_pipeline = x_model_pipeline
        self._enable_outputs_caching = enable_outputs_caching

        if self._enable_outputs_caching:
            assert cache_dir is not None, (
                "cache_dir must be provided if caching is enabled"
            )
            self._cacher = ExplanationStateCacher(
                cache_dir=cache_dir, config=x_model_pipeline.config
            )

            logger.info("Explanation caching enabled.")
            logger.info(f"Storing outputs to file = {self._cacher.file_path}")

    @property
    def name(self) -> str:
        return "explanation"

    def _compute_explanations(self, batch: TensorDataModel) -> BatchExplanationState:
        """Compute explanations for a batch of data."""
        # Collate and move to device
        self._x_model_pipeline.ops.eval()
        with torch.no_grad():
            return self._x_model_pipeline.explanation_step(batch=batch)

    def _combine_outputs(
        self,
        batch: list[TensorDataModel],
        cached_outputs: dict[str, Any],
        missing_batch: list[TensorDataModel],
        new_outputs: list[SampleExplanationState],
    ) -> BatchExplanationState:
        """Combine cached and newly computed outputs in original batch order."""
        new_states_lookup = {
            sample.metadata.sample_id: output
            for sample, output in zip(missing_batch, new_outputs, strict=True)
        }

        combined_outputs = []
        for data_model in batch:
            sample_id = data_model.metadata.sample_id
            if sample_id in cached_outputs:
                combined_outputs.append(cached_outputs[sample_id])
            else:
                combined_outputs.append(new_states_lookup[sample_id])

        return BatchExplanationState.fromlist(combined_outputs)

    def __call__(
        self, engine: Engine, batch: list[TensorDataModel]
    ) -> BatchExplanationState:
        """Process batch with optional caching."""
        if not self._enable_outputs_caching:
            collated_batch = batch[0].batch(batch).ops.to(self._device)
            return self._compute_explanations(collated_batch)

        # Get cached outputs and identify missing samples
        missing_batch = []
        done_samples = 0
        for sample in batch:
            sample_id = sample.metadata.sample_id
            if not self._cacher.sample_exists(sample_id):
                missing_batch.append(sample)
            else:
                done_samples += 1

        logger.debug(
            f"Found {done_samples} / {len(batch)} samples in cache. "
            f"{len(missing_batch)} samples missing."
        )

        # Cache new outputs
        if len(missing_batch) > 0:
            logger.debug(
                f"Computing explanations for {len(missing_batch)} missing samples."
            )
            collated_missing_batch = (
                missing_batch[0].batch(missing_batch).ops.to(self._device)
            )

            # Compute explanations for missing samples
            explanation_states = self._compute_explanations(
                collated_missing_batch
            ).tolist()

            for sample, state in zip(missing_batch, explanation_states, strict=True):
                sample_id = sample.metadata.sample_id
                self._cacher.save_sample(data=state)
                logger.debug(f"Saved explanation to cache for sample_id: {sample_id}")

        # Load all outputs from cache
        explanation_states = []
        for sample in batch:
            sample_id = sample.metadata.sample_id
            cached_state = self._cacher.load_sample(sample_id)

            # for sanity check always make sure the key matches
            assert sample_id == cached_state.sample_id, (
                f"Sample ID mismatch: expected {sample_id}, got {cached_state.sample_id}"
            )

            explanation_states.append(cached_state)

        return BatchExplanationState.fromlist(explanation_states)
