from __future__ import annotations

from collections.abc import Callable

from atria_metrics.core import METRIC, MetricConfig
from atria_metrics.core.classification import _output_transform


@METRIC.register("accuracy")
class AccuracyMetricConfig(MetricConfig):
    module_path: str | None = "ignite.metrics.Accuracy"
    output_transform: Callable = _output_transform
    is_multilabel: bool = False
    skip_unrolling: bool = False


@METRIC.register("precision")
class PrecisionMetricConfig(MetricConfig):
    module_path: str | None = "ignite.metrics.Precision"
    output_transform: Callable = _output_transform
    average: bool = True
    skip_unrolling: bool = False


@METRIC.register("recall")
class RecallMetricConfig(MetricConfig):
    module_path: str | None = "ignite.metrics.Recall"
    output_transform: Callable = _output_transform
    average: bool = True
    skip_unrolling: bool = False


@METRIC.register("confusion_matrix")
class ConfusionMatrixMetricConfig(MetricConfig):
    module_path: str | None = "ignite.metrics.ConfusionMatrix"
    average: str = "recall"
    skip_unrolling: bool = False


@METRIC.register("f1_score")
class F1ScoreMetricConfig(MetricConfig):
    module_path: str | None = (
        "atria_metrics.instance_classification.ext_modules.f1_score"
    )
    output_transform: Callable = _output_transform
