from collections.abc import Callable

import torch
from ignite.metrics import Metric

from atria_metrics.core.entity_labeling.layout_output_transform import _output_transform
from atria_metrics.core.entity_labeling.layout_precision import LayoutPrecision
from atria_metrics.core.entity_labeling.layout_recall import LayoutRecall


def layout_f1(
    output_transform: Callable = _output_transform,
    device: str | torch.device = "cpu",
    average: bool = False,
) -> Metric:
    precision = LayoutPrecision(
        average=False, output_transform=output_transform, device=device
    )
    recall = LayoutRecall(
        average=False, output_transform=output_transform, device=device
    )
    if average == "macro":
        return (precision * recall * 2 / (precision + recall)).mean()
    elif average == "micro":
        return precision.sum() * recall.sum() * 2 / (precision.sum() + recall.sum())
    else:
        return precision * recall * 2 / (precision + recall)
