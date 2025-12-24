"""TorchXAI metric implementation for model explanation evaluation."""

from __future__ import annotations

import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, Generic

import torch
from atria_registry._module_base import ConfigurableModule
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced
from torchxai.explainers import Explainer
from torchxai.ignite._utilities import get_logger

from atria_insights.data_types._explanation_inputs import BatchExplanationInputs
from atria_insights.data_types._explanation_state import MultiTargetBatchExplanation
from atria_insights.data_types._metric_data import BatchMetricData
from atria_insights.engines._explanation_step import ExplanationStepOutput
from atria_insights.explainability_metrics._base import T_ExplainabilityMetricConfig
from atria_insights.storage.sample_cache_managers._metric_data_cacher import (
    MetricDataCacher,
)
from atria_insights.utilities._common import (
    _map_tensor_dicts_to_tuples,
    _map_tensor_tuples_to_keys,
)

logger = get_logger(__name__)


class ExplainabilityMetric(
    Metric,
    ConfigurableModule[T_ExplainabilityMetricConfig],
    Generic[T_ExplainabilityMetricConfig],
):
    __abstract__ = True

    def __init__(
        self,
        model: torch.nn.Module,
        explainer: Explainer,
        config: T_ExplainabilityMetricConfig | None = None,
        device="cpu",
        persist_to_disk: bool = True,
        cache_dir: str | Path | None = None,
    ):
        ConfigurableModule.__init__(self, config=config)

        self._model = model

        self._results = []
        self._num_examples = 0

        # baseline generator
        self._explainer = explainer
        self._baselines_generator = self.config.baselines_generator.build(model=model)
        self._feature_segmentor = self.config.feature_segmentor.build()

        # cache to disk
        self._persist_to_disk = persist_to_disk
        if self._persist_to_disk:
            assert cache_dir is not None, (
                "cache_dir must be provided if caching is enabled"
            )
            self._cacher = MetricDataCacher(cache_dir=cache_dir, config=self.config)

            logger.info("Explanation caching enabled.")
            logger.info(f"Storing outputs to file = {self._cacher.file_path}")

        # assert models match
        assert self._explainer._model == self._model, (
            "Explainer model does not match the metric model."
        )
        assert self._baselines_generator._model == self._model, (
            "Baseline generator model does not match the metric model."
        )

        Metric.__init__(self, output_transform=lambda x: x, device=device)

    def _prepare_baselines(
        self, explanation_inputs: BatchExplanationInputs
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        print("explanation_inputs", explanation_inputs)
        if isinstance(explanation_inputs.inputs, tuple):
            assert explanation_inputs.feature_keys is not None, (
                "Feature keys must be provided when inputs are given as tuple"
            )
            inputs = _map_tensor_tuples_to_keys(
                explanation_inputs.inputs, explanation_inputs.feature_keys
            )
            baselines = self._baselines_generator(inputs=inputs)
            return _map_tensor_dicts_to_tuples(
                baselines, keys=explanation_inputs.feature_keys
            )
        else:
            assert isinstance(explanation_inputs.inputs, torch.Tensor), (
                f"Unexpected type for inputs: {type(explanation_inputs.inputs)}"
            )
            baselines = self._baselines_generator(inputs=explanation_inputs.inputs)
            assert isinstance(baselines, torch.Tensor), (
                f"Unexpected type for baselines: {type(baselines)}"
            )
            return baselines

    def _prepare_feature_mask(self, explanation_inputs: BatchExplanationInputs):
        if isinstance(explanation_inputs.inputs, tuple):
            assert explanation_inputs.feature_keys is not None, (
                "Feature keys must be provided when inputs are given as tuple"
            )
            inputs = _map_tensor_tuples_to_keys(
                explanation_inputs.inputs, explanation_inputs.feature_keys
            )
            feature_mask = self._feature_segmentor(inputs=inputs)
            return _map_tensor_dicts_to_tuples(
                feature_mask, keys=explanation_inputs.feature_keys
            )
        else:
            assert isinstance(explanation_inputs.inputs, torch.Tensor), (
                f"Unexpected type for inputs: {type(explanation_inputs.inputs)}"
            )
            feature_mask = self._feature_segmentor(inputs=explanation_inputs.inputs)
            assert isinstance(feature_mask, torch.Tensor), (
                f"Unexpected type for feature mask: {type(feature_mask)}"
            )
            return feature_mask

    @property
    def name(self) -> str:
        """Return the name of the metric."""
        return self.__class__.__name__

    @reinit__is_reduced
    def reset(self):
        """Reset internal state (called at the start of every epoch)."""
        self._results = []
        self._num_examples = 0
        super().reset()

    @abstractmethod
    def _update(
        self,
        explanation_inputs: BatchExplanationInputs,
        explanations: tuple[torch.Tensor, ...] | list[tuple[torch.Tensor, ...]],
    ) -> dict[str, Any]:
        """Execute the metric function. Must be implemented by subclasses."""
        pass

    def _load_from_disk(self, sample_ids: list[str]) -> dict[str, torch.Tensor]:
        """Load metric data from disk cache."""
        # load full batch from cache
        batch_metric_data = []
        for sample_id in sample_ids:
            cached_data = self._cacher.load_sample(sample_id)
            batch_metric_data.append(cached_data)
        loaded_metric_data = BatchMetricData.fromlist(batch_metric_data)

        logger.debug(
            f"Loaded cached metric data for full batch of size {len(sample_ids)} from disk."
        )

        return loaded_metric_data.data

    @reinit__is_reduced
    def update(self, explanation_step_output: ExplanationStepOutput) -> None:
        """
        Update internal state with output from engine.
        output_transform must return dict with key 'metric_kwargs'.
        """
        if self._persist_to_disk:
            # check if full batch is already done
            is_batch_done = True
            for sample_id in explanation_step_output.explanation_inputs.sample_id:
                if not self._cacher.sample_exists(sample_id):
                    is_batch_done = False
                    break

            if is_batch_done:
                # load full batch from cache
                logger.debug(
                    f"Found cached explanations for full batch of size {len(explanation_step_output.explanation_inputs.sample_id)} from disk."
                )
                data = self._load_from_disk(
                    sample_ids=explanation_step_output.explanation_inputs.sample_id
                )
                self._results.append(data)
                self._num_examples += (
                    explanation_step_output.explanation_inputs.batch_size
                )
                return

        # Measure execution time
        start_time = time.time()

        # Compute metric
        # put items to device
        explanation_inputs = explanation_step_output.explanation_inputs.to_device(
            self._device
        )
        explanation_state = explanation_step_output.explanation_state.to_device(
            self._device
        )

        # convert explanations to list if multi-target
        if isinstance(explanation_state.explanations, MultiTargetBatchExplanation):
            explanations = [e.value for e in explanation_state.explanations.value]
        else:
            explanations = explanation_state.explanations.value

        # compute metric
        metric_output = self._update(
            explanation_inputs=explanation_inputs, explanations=explanations
        )

        # Measure end time
        end_time = time.time()

        # store execution time per sample
        batch_exec_time = torch.tensor(end_time - start_time, requires_grad=False)
        sample_exec_time = torch.stack(
            [
                batch_exec_time / explanation_inputs.batch_size
                for _ in range(explanation_inputs.batch_size)
            ]
        )

        metric_data = BatchMetricData(
            sample_id=explanation_inputs.sample_id,
            data={**metric_output, "sample_exec_time": sample_exec_time},
        )

        # save to disk
        if self._persist_to_disk:
            for sample_metric_data in metric_data.tolist():
                self._cacher.save_sample(sample_metric_data)

        # Accumulate results
        self._num_examples += explanation_inputs.batch_size
        self._results.append(metric_data.data)

    # -----------------------------------------------------------------
    def compute(self):
        """Compute final metric from accumulated state."""
        return self._results
