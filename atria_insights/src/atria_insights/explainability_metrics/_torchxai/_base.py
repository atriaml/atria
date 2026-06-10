"""TorchXAI metric implementation for model explanation evaluation."""

from __future__ import annotations

import time
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Mapping
from typing import ClassVar, Generic

import numpy as np
import torch
from atria_logger import get_logger
from atria_registry._module_base import ConfigurableModule
from ignite.engine import Engine
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced
from pydantic import BaseModel, field_validator
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


class BatchMetricOuptut(BaseModel):
    key: str
    value: list[float] | list[list[float]]


class MultiTargetBatchMetricOuptut(BaseModel):
    key: str
    value: list[list[float]] | list[list[list[float]]]

    # hanlde multi target transform
    @field_validator("value", mode="after")
    def _transpose_value(
        value: list[list[float]] | list[list[list[float]]],
    ) -> list[MultiTargetTimedBatchMetricOuptut]:
        print("value", len(value), len(value[0]))
        transformed_value = list(map(list, zip(*value, strict=True)))
        print("transformed_value", len(transformed_value), len(transformed_value[0]))
        return transformed_value


class TimedBatchMetricOuptut(BatchMetricOuptut):
    key: str
    value: list[float] | list[list[float]]
    exec_time_ms: list[float]


class MultiTargetTimedBatchMetricOuptut(BatchMetricOuptut):
    key: str
    value: list[list[float]] | list[list[list[float]]]
    exec_time_ms: list[float]


class ExplainabilityMetric(
    Metric,
    ConfigurableModule[T_ExplainabilityMetricConfig],
    Generic[T_ExplainabilityMetricConfig],
):
    __abstract__ = True
    _score_keys: ClassVar[list[str]] = []

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
    ):
        Metric.__init__(self, output_transform=lambda x: x, device=device)
        ConfigurableModule.__init__(self, config=config)

        self._model = model

        self._results = []
        self._num_examples = 0

        # baseline generator
        self._explainer = explainer

        # cache to disk
        self._metric_unique_name = config.type.replace("/", ".")
        self._metric_key = self._metric_unique_name + "-" + config.hash
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
    ) -> list[BatchMetricOuptut | MultiTargetBatchMetricOuptut]:
        """Execute the metric function. Must be implemented by subclasses."""
        pass

    def _get_sample_key(self, sample_id: str):
        return "/".join([self._metric_key, sample_id])

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

    def _timed_update(
        self, explanation_step_output: ExplanationStepOutput
    ) -> list[TimedBatchMetricOuptut]:
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
        batch_metric_outputs = self._update(
            explanation_inputs=explanation_inputs, explanations=explanations
        )

        # Measure end time
        end_time = time.time()

        # store execution time per sample
        batch_exec_time = [
            (end_time - start_time) * 1000 / explanation_inputs.batch_size
            for _ in range(explanation_inputs.batch_size)
        ]

        cls = (
            TimedBatchMetricOuptut
            if not explanation_inputs.is_multi_target
            else MultiTargetTimedBatchMetricOuptut
        )
        timed_batch_metric_output = [
            cls(
                key=batch_metric_output.key,
                value=batch_metric_output.value,
                exec_time_ms=batch_exec_time,
            )
            for batch_metric_output in batch_metric_outputs
        ]
        return timed_batch_metric_output

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
                if not self._cacher.sample_exists(self._get_sample_key(sample_id)):
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
                    self._results.append(data)
                    self._num_examples += (
                        explanation_step_output.explanation_inputs.batch_size
                    )
                    print("results loaded from cache", data)
                    return
                except Exception as e:
                    logger.warning(
                        f"Failed to load metric data from disk cache for batch. Recomputing metric. Error: {e}"
                    )
                    is_batch_done = False

        logger.debug(f"Computing metric {self.name}.")

        timed_batch_metric_outputs = self._timed_update(
            explanation_step_output=explanation_step_output
        )

        # flatten the batch metric outputs
        metric_output = {}
        for timed_batch_metric_output in timed_batch_metric_outputs:
            metric_output[f"metric/{timed_batch_metric_output.key}/score"] = (
                timed_batch_metric_output.value
            )
            metric_output[f"metric/{timed_batch_metric_output.key}/exec_time_ms"] = (
                timed_batch_metric_output.exec_time_ms
            )

        metric_data = BatchMetricData(
            sample_id=explanation_step_output.explanation_inputs.sample_id,
            data={**metric_output},
            config=self.config.model_dump(),
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
        self._num_examples += explanation_step_output.explanation_inputs.batch_size
        self._results.append(metric_data.data)
        print("results loaded directly", metric_data.data)

    def compute(self) -> dict:
        if not self._results:
            return {}

        def flatten(lst):
            return [x for xs in lst for x in xs]

        accumulated = defaultdict(list)
        for result in self._results:
            for key, values in result.items():
                if isinstance(values[0], list):
                    accumulated[key].extend(flatten(values))
                else:
                    accumulated[key].extend(values)
        mean_results = {key: np.mean(values) for key, values in accumulated.items()}
        return mean_results

    def completed(self, engine: Engine, name: str) -> None:
        result = self.compute()

        # we ignore the name of the attachement
        if isinstance(result, Mapping):
            if name in result.keys():
                raise ValueError(
                    f"Argument name '{name}' is conflicting with mapping keys: {list(result.keys())}"
                )

            for key, value in result.items():
                engine.state.metrics[key] = value
        else:
            if isinstance(result, torch.Tensor):
                if len(result.size()) == 0:
                    result = result.item()
                elif "cpu" not in result.device.type:
                    result = result.cpu()

            engine.state.metrics[self._metric_unique_name] = result

    def __str__(self):
        return f"{self.__class__.__name__}(config={self.config})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
