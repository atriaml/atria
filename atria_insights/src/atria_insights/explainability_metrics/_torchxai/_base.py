"""TorchXAI metric implementation for model explanation evaluation."""

from __future__ import annotations

import time
from abc import abstractmethod
from typing import Any, Generic

import torch
from atria_logger import get_logger
from atria_registry._module_base import ConfigurableModule
from ignite.engine import Engine
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced
from torchxai.data_types import ExplanationTarget
from torchxai.explainers import Explainer

from atria_insights.data_types._explanation_inputs import BatchExplanationInputs
from atria_insights.data_types._explanation_state import MultiTargetBatchExplanation
from atria_insights.data_types._metric_data import BatchMetricData
from atria_insights.data_types._targets import BatchExplanationTarget
from atria_insights.engines._events import MetricUpdateEvents
from atria_insights.engines._explanation_step import ExplanationStepOutput
from atria_insights.explainability_metrics._base import T_ExplainabilityMetricConfig
from atria_insights.storage.sample_cache_managers._metric_data_cacher import (
    MetricDataCacher,
)

logger = get_logger(__name__)


class ExplainabilityMetric(
    Metric,
    ConfigurableModule[T_ExplainabilityMetricConfig],
    Generic[T_ExplainabilityMetricConfig],
):
    __abstract__ = True

    @property
    def name(self):
        return self.__class__.__name__

    def __init__(
        self,
        model: torch.nn.Module,
        explainer: Explainer,
        config: T_ExplainabilityMetricConfig | None = None,
        device="cpu",
        cacher: MetricDataCacher | None = None,
        metric_key: str | None = None,
    ):
        Metric.__init__(self, output_transform=lambda x: x, device=device)
        ConfigurableModule.__init__(self, config=config)

        self._model = model

        self._results = []
        self._num_examples = 0

        # baseline generator
        self._explainer = explainer

        # cache to disk
        self._metric_key = metric_key
        self._cacher = cacher

        # assert models match
        assert self._explainer._model == self._model, (
            "Explainer model does not match the metric model."
        )

    @property
    def config(self) -> T_ExplainabilityMetricConfig:
        """Return the configuration of the metric."""
        return super().config

    def _map_target(
        self, target: BatchExplanationTarget | list[BatchExplanationTarget] | None
    ) -> ExplanationTarget | list[ExplanationTarget]:
        if target is None:
            return ExplanationTarget.from_raw_input(None)
        if isinstance(target, BatchExplanationTarget):
            return ExplanationTarget.from_raw_input(target.value)
        elif isinstance(target, list):
            return [ExplanationTarget.from_raw_input(t.value) for t in target]
        else:
            raise ValueError(
                "Target must be of type BatchExplanationTarget, list of BatchExplanationTarget, or None."
            )

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

    def _get_sample_key(self, sample_id: str):
        return "-".join([self._metric_key, sample_id])

    def _load_from_disk(self, sample_ids: list[str]) -> dict[str, torch.Tensor]:
        """Load metric data from disk cache."""
        # load full batch from cache
        batch_metric_data = []
        for sample_id in sample_ids:
            sample_key = self._get_sample_key(sample_id)
            cached_data = self._cacher.load_sample(sample_key)
            batch_metric_data.append(cached_data)
        loaded_metric_data = BatchMetricData.fromlist(batch_metric_data)

        logger.debug(
            f"Loaded cached metric data for full batch of size {len(sample_ids)} from disk."
        )

        return loaded_metric_data.data

    @torch.no_grad()
    def iteration_completed(self, engine: Engine) -> None:
        engine.state.x_metric_started = self.name
        engine.fire_event(MetricUpdateEvents.X_METRIC_STARTED)
        super().iteration_completed(engine)
        engine.state.x_metric_completed = self.name
        engine.fire_event(MetricUpdateEvents.X_METRIC_COMPLETED)

    @reinit__is_reduced
    def update(self, explanation_step_output: ExplanationStepOutput) -> None:
        """
        Update internal state with output from engine.
        output_transform must return dict with key 'metric_kwargs'.
        """
        if self._cacher is not None:
            # check if full batch is already done
            is_batch_done = True
            for sample_id in explanation_step_output.explanation_inputs.sample_id:
                if not self._cacher.sample_exists(sample_id):
                    is_batch_done = False
                    break

            if is_batch_done:
                # load full batch from cache
                logger.debug(
                    f"Found cached metric for full batch of size {len(explanation_step_output.explanation_inputs.sample_id)} from disk."
                )
                try:
                    data = self._load_from_disk(
                        sample_ids=explanation_step_output.explanation_inputs.sample_id
                    )
                    # logger.info("Metric data loaded.")
                    self._results.append(data)
                    self._num_examples += (
                        explanation_step_output.explanation_inputs.batch_size
                    )
                    return
                except Exception as e:
                    logger.warning(
                        f"Failed to load metric data from disk cache for batch. Recomputing metric. Error: {e}"
                    )
                    is_batch_done = False

        logger.debug(f"Computing metric {self.name}.")

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

        logger.info(
            f"Computing metric {self.name} for batch size {explanation_inputs.batch_size}."
        )
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

        # if it is multitarget each key, value would be key, list {batch values} so we transpose it
        if explanation_inputs.is_multi_target:
            n_targets = len(explanation_inputs.target)
            logger.debug(
                f"Transposing multi-target metric output for {n_targets} targets."
            )
            for key, value in metric_output.items():
                assert isinstance(value, list), (
                    f"Expected list for multi-target metric output, got {type(value)}"
                )
                if isinstance(value[0], torch.Tensor):
                    metric_output[key] = torch.stack(value).transpose(0, 1)
                    assert (
                        metric_output[key].shape[0] == explanation_inputs.batch_size
                    ), (
                        f"Expected shape[0] to be batch size {explanation_inputs.batch_size}, got {metric_output[key].shape[0]}"
                    )
                    assert metric_output[key].shape[1] == n_targets, (
                        f"Expected shape[1] to be number of targets {len(value)}, got {metric_output[key].shape[1]}"
                    )
                else:
                    metric_output[key] = list(map(list, zip(*value, strict=True)))
                    assert len(metric_output[key]) == explanation_inputs.batch_size, (
                        f"Expected length to be batch size {explanation_inputs.batch_size}, got {len(metric_output[key])}"
                    )
                    assert all(
                        len(metric_output[key][i]) == n_targets
                        for i in range(explanation_inputs.batch_size)
                    ), f"Expected inner length to be number of targets {n_targets}"

                # recursively convert numpy arrays to tensors if any
                def _convert_to_tensor(item):
                    import numpy as np

                    if isinstance(item, np.ndarray):
                        return torch.tensor(item)
                    elif isinstance(item, list):
                        return [_convert_to_tensor(i) for i in item]
                    elif isinstance(item, dict):
                        return {k: _convert_to_tensor(v) for k, v in item.items()}
                    else:
                        return item

                metric_output[key] = _convert_to_tensor(metric_output[key])

            logger.debug(f"Transposed metric output: {metric_output}")

        metric_data = BatchMetricData(
            sample_id=explanation_inputs.sample_id,
            data={**metric_output, "sample_exec_time": sample_exec_time},
        )

        logger.info(
            f"Metric data computed: {self.name} for batch size {metric_data.batch_size}"
        )

        # save to disk
        if self._cacher is not None:
            for sample_metric_data in metric_data.tolist():
                sample_metric_data = sample_metric_data.model_copy(
                    update={
                        "sample_id": self._get_sample_key(sample_metric_data.sample_id)
                    }
                )
                self._cacher.save_sample(sample_metric_data)

        # Accumulate results
        self._num_examples += explanation_inputs.batch_size
        self._results.append(metric_data.data)

    # -----------------------------------------------------------------
    def compute(self):
        """Compute final metric from accumulated state."""
        return self._results

    def __str__(self):
        return f"{self.__class__.__name__}(config={self.config})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
