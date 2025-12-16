"""TorchXAI metric implementation for model explanation evaluation."""

from __future__ import annotations

import time
from abc import abstractmethod
from typing import Any

import torch
from atria_insights.src.atria_insights.core.data_types import (
    ModelExplainerOutput,
    MultiTargetModelExplainerOutput,
)
from atria_logger import get_logger
from ignite.metrics import Metric
from ignite.metrics.metric import Metric, reinit__is_reduced
from ignite.utils import apply_to_tensor
from torch.cuda.amp.autocast_mode import autocast

logger = get_logger(__name__)


class TorchXAIMetricBase(Metric):
    def __init__(self, model: torch.nn.Module, with_amp: bool = False, device="cpu"):
        self._model = model
        self._with_amp = with_amp

        self._results = []
        self._num_examples = 0

        super().__init__(output_transform=lambda x: x, device=device)

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
        self, model_output: ModelExplainerOutput, is_multi_target: bool = False
    ) -> Any:
        """Execute the metric function. Must be implemented by subclasses."""
        pass

    @reinit__is_reduced
    def update(self, model_output: ModelExplainerOutput) -> None:
        """
        Update internal state with output from engine.
        output_transform must return dict with key 'metric_kwargs'.
        """
        logger.info(f"Updating metric: {self.name}")

        # Measure execution time
        start_time = time.time()

        # Compute metric
        with autocast(enabled=self._with_amp):
            metric_output = self._update(
                model_output=model_output,
                is_multi_target=isinstance(
                    model_output, MultiTargetModelExplainerOutput
                ),
                return_dict=True,
            )

        # Measure end time
        end_time = time.time()

        # detach tensors to move them to CPU
        metric_output = apply_to_tensor(
            metric_output, lambda tensor: tensor.detach().cpu()
        )

        # store execution time per sample
        batch_exec_time = torch.tensor(end_time - start_time, requires_grad=False)
        batch_size = len(model_output.sample_id)
        sample_exec_time = (
            torch.stack(
                [
                    batch_exec_time / batch_size
                    for _ in range(len(model_output.sample_id))
                ]
            ),
        )

        # Accumulate results
        self._num_examples += batch_size
        self._results.append({**metric_output, "sample_exec_time": sample_exec_time})  # type: ignore

    # -----------------------------------------------------------------
    def compute(self):
        """Compute final metric from accumulated state."""
        if self._num_examples == 0:
            return {}
        return self._results
