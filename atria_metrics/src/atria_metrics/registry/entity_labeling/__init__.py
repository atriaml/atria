from __future__ import annotations

from collections.abc import Callable

from atria_metrics.core import METRIC
from atria_metrics.core.classification import _output_transform
from atria_metrics.core.entity_labeling import SeqEvalMetricConfig


@METRIC.register("seqeval_accuracy_score")
class SeqEvalAccuracyMetricConfig(SeqEvalMetricConfig):
    name: str = "seqeval_accuracy_score"
    module_path: str | None = "seqeval.metrics.accuracy_score"
    output_transform: Callable = _output_transform
    skip_unrolling: bool = False


@METRIC.register("seqeval_precision_score")
class SeqEvalPrecisionMetricConfig(SeqEvalMetricConfig):
    name: str = "seqeval_precision_score"
    module_path: str | None = "seqeval.metrics.precision_score"
    output_transform: Callable = _output_transform
    scheme: str = "IOB2"


@METRIC.register("seqeval_recall_score")
class SeqEvalRecallMetricConfig(SeqEvalMetricConfig):
    name: str = "seqeval_recall_score"
    module_path: str | None = "seqeval.metrics.recall_score"
    output_transform: Callable = _output_transform
    scheme: str = "IOB2"


@METRIC.register("seqeval_f1_score")
class SeqEvalF1ScoreMetricConfig(SeqEvalMetricConfig):
    name: str = "seqeval_f1_score"
    module_path: str | None = "seqeval.metrics.f1_score"
    output_transform: Callable = _output_transform
    scheme: str = "IOB2"
    skip_unrolling: bool = False


@METRIC.register("seqeval_classification_report")
class SeqEvalClassificationReportMetricConfig(SeqEvalMetricConfig):
    name: str = "seqeval_classification_report"
    module_path: str | None = "seqeval.metrics.classification_report"
    output_transform: Callable = _output_transform
    scheme: str = "IOB2"
