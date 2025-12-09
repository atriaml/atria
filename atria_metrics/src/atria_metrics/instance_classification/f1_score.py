from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from atria_metrics.instance_classification.output_transforms import _output_transform
from atria_metrics.registry import METRIC

if TYPE_CHECKING:
    from ignite.metrics import Metric


@METRIC.register("f1_score", output_transform=_output_transform)
def f1_score(output_transform: Callable, device: str = "cpu") -> Metric:
    from ignite.metrics import Precision, Recall

    # use ignite arthematics of metrics to compute f1 score
    # unnecessary complication
    precision = Precision(
        average=False, output_transform=output_transform, device=device
    )
    recall = Recall(average=False, output_transform=output_transform, device=device)
    return (precision * recall * 2 / (precision + recall)).mean()
