from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

from atria_models.core.types.model_outputs import TokenClassificationModelOutput
from atria_registry._utilities import _resolve_module_from_path

from atria_metrics.core import METRIC, MetricConfig
from atria_metrics.core._base import MetricConfig
from atria_metrics.core.classification import _output_transform
from atria_metrics.registry import METRIC

if TYPE_CHECKING:
    import torch
    from ignite.metrics import Metric


@METRIC.register("seqeval_accuracy_score")
class SeqEvalAccuracyMetricConfig(SeqEvalMetricConfig):
    module_path: str | None = "seqeval.metrics.accuracy_score"
    output_transform: Callable = _output_transform
    skip_unrolling: bool = False


@METRIC.register("seqeval_precision_score")
class SeqEvalPrecisionMetricConfig(SeqEvalMetricConfig):
    module_path: str | None = "seqeval.metrics.precision_score"
    output_transform: Callable = _output_transform
    scheme: str = "IOB2"


@METRIC.register("seqeval_recall_score")
class SeqEvalRecallMetricConfig(SeqEvalMetricConfig):
    module_path: str | None = "seqeval.metrics.recall_score"
    output_transform: Callable = _output_transform
    scheme: str = "IOB2"


@METRIC.register("seqeval_f1_score")
class SeqEvalF1ScoreMetricConfig(SeqEvalMetricConfig):
    module_path: str | None = "seqeval.metrics.f1_score"
    output_transform: Callable = _output_transform
    scheme: str = "IOB2"
    skip_unrolling: bool = False


@METRIC.register("seqeval_classification_report")
class SeqEvalClassificationReportMetricConfig(SeqEvalMetricConfig):
    module_path: str | None = (
        "atria_metrics.token_classification.seqeval.seqeval_metric"
    )
    output_transform: Callable = _output_transform
    scheme: str = "IOB2"
