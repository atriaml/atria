from collections.abc import Callable

import torch
from ignite.metrics import Metric

from atria_metrics.layout.output_transforms import _output_transform
from atria_metrics.layout.precision import LayoutPrecision
from atria_metrics.layout.recall import LayoutRecall
from atria_metrics.registry import METRIC


@METRIC.register("layout_f1", output_transform=_output_transform)
def f1_score(output_transform: Callable, device: str | torch.device = "cpu") -> Metric:
    # use ignite arthematics of metrics to compute f1 score
    # unnecessary complication
    precision = LayoutPrecision(
        average=False, output_transform=output_transform, device=device
    )
    recall = LayoutRecall(
        average=False, output_transform=output_transform, device=device
    )
    return (precision * recall * 2 / (precision + recall)).mean()
