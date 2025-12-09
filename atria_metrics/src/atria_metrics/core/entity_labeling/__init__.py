from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

from atria_registry._utilities import _resolve_module_from_path

from atria_metrics.core import MetricConfig
from atria_metrics.core.classification import _output_transform

if TYPE_CHECKING:
    import torch
    from ignite.metrics import Metric


class SeqEvalMetricConfig(MetricConfig):
    @property
    def kwargs(self) -> dict[str, Any]:
        return self.model_dump(exclude={"module_path", "output_transform"})

    def build(  # type: ignore
        self, device: str | torch.device = "cpu", num_classes: int | None = None
    ) -> Metric:
        from atria_metrics.core._epoch_dict_metric import EpochDictMetric

        assert self.module_path is not None, (
            "module_path must be set to build the module."
        )
        metric_func = _resolve_module_from_path(self.module_path)
        return EpochDictMetric(
            compute_fn=partial(metric_func, **self.kwargs),
            output_transform=_output_transform,
            device=device,
        )
