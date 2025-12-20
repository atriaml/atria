from __future__ import annotations

from typing import Literal

from atria_metrics.core import METRICS, MetricConfig


@METRICS.register("seqeval")
class SeqEvalMetricConfig(MetricConfig):
    name: Literal["seqeval"] = "seqeval"
    module_path: str | None = (
        "atria_metrics.core.entity_labeling.seqeval_metric.SeqEvalMetric"
    )
    scheme: str = "IOB2"


@METRICS.register("layout_precision")
class LayoutPrecisionMetricConfig(MetricConfig):
    name: Literal["layout_precision"] = "layout_precision"
    module_path: str | None = (
        "atria_metrics.core.entity_labeling.layout_precision.LayoutPrecision"
    )
    average: bool | str = False


@METRICS.register("layout_recall")
class LayoutRecallMetricConfig(MetricConfig):
    name: Literal["layout_recall"] = "layout_recall"
    module_path: str | None = (
        "atria_metrics.core.entity_labeling.layout_recall.LayoutRecall"
    )
    average: bool | str = False


@METRICS.register("layout_f1")
class LayoutF1MetricConfig(MetricConfig):
    name: Literal["layout_f1"] = "layout_f1"
    module_path: str | None = "atria_metrics.core.entity_labeling.layout_f1.layout_f1"
    average: bool | str = False


@METRICS.register("layout_precision_macro")
class LayoutPrecisionMacroMetricConfig(LayoutPrecisionMetricConfig):
    name: Literal["layout_precision_macro"] = "layout_precision_macro"
    average: bool | str = "macro"


@METRICS.register("layout_recall_macro")
class LayoutRecallMacroMetricConfig(LayoutRecallMetricConfig):
    name: Literal["layout_recall_macro"] = "layout_recall_macro"
    average: bool | str = "macro"


@METRICS.register("layout_f1_macro")
class LayoutF1MacroMetricConfig(LayoutF1MetricConfig):
    name: Literal["layout_f1_macro"] = "layout_f1_macro"
    average: bool | str = "macro"
