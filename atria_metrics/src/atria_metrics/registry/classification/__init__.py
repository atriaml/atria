from __future__ import annotations

from typing import Literal

from atria_metrics.core import METRIC, MetricConfig
from atria_metrics.core.classification import ClassificationMetricConfig


@METRIC.register("accuracy")
class AccuracyMetricConfig(ClassificationMetricConfig):
    name: Literal["accuracy"] = "accuracy"
    module_path: str | None = "ignite.metrics.Accuracy"
    is_multilabel: bool = False
    skip_unrolling: bool = False


@METRIC.register("precision")
class PrecisionMetricConfig(ClassificationMetricConfig):
    name: Literal["precision"] = "precision"
    module_path: str | None = "ignite.metrics.Precision"
    average: bool = True
    skip_unrolling: bool = False


@METRIC.register("recall")
class RecallMetricConfig(ClassificationMetricConfig):
    name: Literal["recall"] = "recall"
    module_path: str | None = "ignite.metrics.Recall"
    average: bool = True
    skip_unrolling: bool = False


@METRIC.register("confusion_matrix")
class ConfusionMatrixMetricConfig(ClassificationMetricConfig):
    name: Literal["confusion_matrix"] = "confusion_matrix"
    module_path: str | None = "ignite.metrics.ConfusionMatrix"
    average: str = "recall"
    skip_unrolling: bool = False


@METRIC.register("f1_score")
class F1ScoreMetricConfig(ClassificationMetricConfig):
    name: Literal["f1_score"] = "f1_score"
    module_path: str | None = "atria_metrics.core.classification.f1_score"
