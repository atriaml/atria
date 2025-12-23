# type: ignore
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from atria_logger import get_logger
from atria_ml.training.engine_steps._base import EngineStep
from atria_transforms.core._data_types._base import TensorDataModel
from ignite.engine import Engine
from pydantic import BaseModel, ConfigDict

from atria_insights.data_types._explanation_inputs import BatchExplanationInputs
from atria_insights.data_types._explanation_state import BatchExplanationState
from atria_insights.storage.sample_cache_managers._explanation_state import (
    ExplanationStateCacher,
)

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
        persist_to_disk: bool = False,
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
        self._persist_to_disk = persist_to_disk

        if self._persist_to_disk:
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

    def _compute_or_load_explanations_for_batch(
        self, batch: TensorDataModel
    ) -> BatchExplanationState:
        """Compute or load explanations for a batch of data."""
        # prepare explanation inputs
        model_outputs, explanation_inputs = (
            self._x_model_pipeline.prepare_explanation_inputs(
                batch=batch[0].batch(batch).ops.to(self._device)
            )
        )

        if self._persist_to_disk:
            # check if full batch is already done
            is_batch_done = True
            for sample_id in explanation_inputs.sample_id:
                if not self._cacher.sample_exists(sample_id):
                    is_batch_done = False
                    break

            if is_batch_done:
                # load full batch from cache
                batch_explanation_state = []
                for sample_id in explanation_inputs.sample_id:
                    cached_state = self._cacher.load_sample(sample_id)
                    batch_explanation_state.append(cached_state)
                loaded_explanation_state = BatchExplanationState.fromlist(
                    batch_explanation_state
                )

                def _validate_loaded_state(
                    loaded_explanation_state: BatchExplanationState,
                    explanation_inputs: BatchExplanationInputs,
                ) -> BatchExplanationState:
                    assert (
                        loaded_explanation_state.sample_id
                        == explanation_inputs.sample_id
                    ), (
                        "Sample IDs do not match between loaded explanation states and explanation inputs."
                    )
                    assert (
                        loaded_explanation_state.target == explanation_inputs.target
                    ), (
                        "Targets do not match between loaded explanation states and explanation inputs."
                    )
                    assert (
                        loaded_explanation_state.feature_keys
                        == explanation_inputs.feature_keys
                    ), (
                        "Feature keys do not match between loaded explanation states and explanation inputs."
                    )
                    assert (
                        loaded_explanation_state.frozen_features
                        == explanation_inputs.frozen_features
                    ), (
                        "Frozen features do not match between loaded explanation states and explanation inputs."
                    )
                    assert torch.allclose(
                        loaded_explanation_state.model_outputs, model_outputs
                    ), (
                        "Model outputs do not match between loaded explanation states and current model outputs."
                    )

                return loaded_explanation_state

        explanations = self._x_model_pipeline.explainer_forward(
            explanation_inputs=explanation_inputs
        )

        # prepare explanation states
        batch_explanation_state = BatchExplanationState(
            sample_id=explanation_inputs.sample_id,
            target=explanation_inputs.target,
            feature_keys=explanation_inputs.feature_keys,
            frozenset_features=explanation_inputs.frozen_features,
            model_outputs=model_outputs,
            explanations=explanations,
        )

        return explanation_inputs, model_outputs, batch_explanation_state

    def _validate_and_load_from_disk(
        self, explanation_inputs: BatchExplanationInputs, model_outputs: torch.Tensor
    ) -> ExplanationStepOutput:
        # load full batch from cache
        explanation_state = []
        for sample_id in explanation_inputs.sample_id:
            cached_state = self._cacher.load_sample(sample_id)
            explanation_state.append(cached_state)

        explanation_state = BatchExplanationState.fromlist(explanation_state)

        assert explanation_state.sample_id == explanation_inputs.sample_id, (
            "Sample IDs do not match between loaded explanation states and explanation inputs."
        )
        assert explanation_state.target == explanation_inputs.target, (
            "Targets do not match between loaded explanation states and explanation inputs."
        )
        assert explanation_state.feature_keys == explanation_inputs.feature_keys, (
            "Feature keys do not match between loaded explanation states and explanation inputs."
        )
        assert (
            explanation_state.frozen_features == explanation_inputs.frozen_features
        ), (
            "Frozen features do not match between loaded explanation states and explanation inputs."
        )
        assert torch.allclose(
            explanation_state.model_outputs.detach().cpu(),
            model_outputs.detach().cpu(),
            atol=1e-4,
        ), (
            "Model outputs do not match between loaded explanation states and current model outputs."
            f"Found {model_outputs.detach().cpu()} =/= {explanation_state.model_outputs.detach().cpu()}"
        )

        return ExplanationStepOutput(
            explanation_inputs=explanation_inputs, explanation_state=explanation_state
        )

    def __call__(
        self, engine: Engine, batch: list[TensorDataModel]
    ) -> BatchExplanationState:
        """Process batch with optional caching."""
        # set model to eval mode
        self._x_model_pipeline.ops.eval()

        """Compute or load explanations for a batch of data."""
        # prepare explanation inputs
        model_outputs, explanation_inputs = (
            self._x_model_pipeline.prepare_explanation_inputs(
                batch=batch[0].batch(batch).ops.to(self._device)
            )
        )

        if self._persist_to_disk:
            # check if full batch is already done
            is_batch_done = True
            for sample_id in explanation_inputs.sample_id:
                if not self._cacher.sample_exists(sample_id):
                    is_batch_done = False
                    break

            if is_batch_done:
                # load full batch from cache
                logger.debug(
                    f"Found cached explanations for full batch of size {len(batch)}. Loading from disk."
                )
                return self._validate_and_load_from_disk(
                    explanation_inputs=explanation_inputs, model_outputs=model_outputs
                )

        explanations = self._x_model_pipeline.explainer_forward(
            explanation_inputs=explanation_inputs
        )

        # prepare explanation states
        explanation_state = BatchExplanationState(
            sample_id=explanation_inputs.sample_id,
            target=explanation_inputs.target,
            feature_keys=explanation_inputs.feature_keys,
            frozen_features=explanation_inputs.frozen_features,
            model_outputs=model_outputs,
            explanations=explanations,
        )

        # save to disk
        if self._persist_to_disk:
            for sample_explanation_state in explanation_state.tolist():
                self._cacher.save_sample(sample_explanation_state)

        return ExplanationStepOutput(
            explanation_inputs=explanation_inputs, explanation_state=explanation_state
        )
