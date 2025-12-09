from collections.abc import Callable

import torch
from atria_core.utilities.imports import _get_parent_module, _resolve_module_from_path

from atria_metrics.registry import METRIC
from atria_metrics.token_classification.output_transforms import _output_transform


def seqeval_classification_metric(
    metric_func: str,
    output_transform: Callable,
    device: str | torch.device = "cpu",
    **kwargs,
):
    from functools import partial

    from atria_metrics.common.epoch_dict_metric import EpochDictMetric

    return EpochDictMetric(
        compute_fn=partial(_resolve_module_from_path(metric_func), **kwargs),
        output_transform=output_transform,
        device=device,
    )


for metric in [
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "classification_report",
]:
    kwargs = {}
    if metric in [
        "precision_score",
        "recall_score",
        "f1_score",
        "classification_report",
    ]:
        kwargs["scheme"] = "IOB2"

    METRIC.register(
        f"seqeval_{metric}",
        metric_func=f"seqeval.metrics.{metric}",
        output_transform=_output_transform,
        device="cpu",
        **kwargs,
    )(_get_parent_module(__name__) + ".seqeval.seqeval_classification_metric")
