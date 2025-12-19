from __future__ import annotations

from atria_metrics.core import METRICS
from atria_metrics.core.entity_labeling import SeqEvalMetricConfig


@METRICS.register("seqeval_accuracy_score")
class SeqEvalAccuracyMetricConfig(SeqEvalMetricConfig):
    name: str = "seqeval_accuracy_score"
    module_path: str | None = "seqeval.metrics.accuracy_score"
    skip_unrolling: bool = False


@METRICS.register("seqeval_precision_score")
class SeqEvalPrecisionMetricConfig(SeqEvalMetricConfig):
    name: str = "seqeval_precision_score"
    module_path: str | None = "seqeval.metrics.precision_score"
    scheme: str = "IOB2"


@METRICS.register("seqeval_recall_score")
class SeqEvalRecallMetricConfig(SeqEvalMetricConfig):
    name: str = "seqeval_recall_score"
    module_path: str | None = "seqeval.metrics.recall_score"
    scheme: str = "IOB2"


@METRICS.register("seqeval_f1_score")
class SeqEvalF1ScoreMetricConfig(SeqEvalMetricConfig):
    name: str = "seqeval_f1_score"
    module_path: str | None = "seqeval.metrics.f1_score"
    scheme: str = "IOB2"
    skip_unrolling: bool = False


@METRICS.register("seqeval_classification_report")
class SeqEvalClassificationReportMetricConfig(SeqEvalMetricConfig):
    name: str = "seqeval_classification_report"
    module_path: str | None = "seqeval.metrics.classification_report"
    scheme: str = "IOB2"
