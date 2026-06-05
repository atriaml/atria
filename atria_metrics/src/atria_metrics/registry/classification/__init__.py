from __future__ import annotations

from typing import ClassVar, Literal

from atria_metrics.core import METRICS, MetricConfig
from atria_metrics.core.classification import ClassificationMetricConfig


@METRICS.register("accuracy")
class AccuracyMetricConfig(ClassificationMetricConfig):
    __module_path__: ClassVar[str] =  "ignite.metrics.Accuracy"
    name: Literal["accuracy"] = "accuracy"
    is_multilabel: bool = False
    skip_unrolling: bool = False


@METRICS.register("precision")
class PrecisionMetricConfig(ClassificationMetricConfig):
    __module_path__: ClassVar[str] =  "ignite.metrics.Precision"
    name: Literal["precision"] = "precision"
    average: bool = True
    skip_unrolling: bool = False


@METRICS.register("recall")
class RecallMetricConfig(ClassificationMetricConfig):
    __module_path__: ClassVar[str] =  "ignite.metrics.Recall"
    name: Literal["recall"] = "recall"
    average: bool = True
    skip_unrolling: bool = False


@METRICS.register("confusion_matrix")
class ConfusionMatrixMetricConfig(ClassificationMetricConfig):
    __module_path__: ClassVar[str] =  "ignite.metrics.ConfusionMatrix"
    name: Literal["confusion_matrix"] = "confusion_matrix"
    average: str = "recall"
    skip_unrolling: bool = False


@METRICS.register("f1_score")
class F1ScoreMetricConfig(ClassificationMetricConfig):
    __module_path__: ClassVar[str] =  "atria_metrics.core.classification.f1_score"
    name: Literal["f1_score"] = "f1_score"
