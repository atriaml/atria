from __future__ import annotations

from typing import ClassVar, Literal

from atria_metrics.core import METRICS, MetricConfig


@METRICS.register("seqeval")
class SeqEvalMetricConfig(MetricConfig):
    __module_path__: ClassVar[str] = (
        "atria_metrics.core.entity_labeling.seqeval_metric.SeqEvalMetric"
    )
    name: Literal["seqeval"] = "seqeval"
    scheme: str = "IOB2"


@METRICS.register("layout_precision")
class LayoutPrecisionMetricConfig(MetricConfig):
    __module_path__: ClassVar[str] = (
        "atria_metrics.core.entity_labeling.layout_precision.LayoutPrecision"
    )
    name: Literal["layout_precision"] = "layout_precision"
    average: bool | str = False


@METRICS.register("layout_recall")
class LayoutRecallMetricConfig(MetricConfig):
    __module_path__: ClassVar[str] = (
        "atria_metrics.core.entity_labeling.layout_recall.LayoutRecall"
    )
    name: Literal["layout_recall"] = "layout_recall"
    average: bool | str = False


@METRICS.register("layout_f1")
class LayoutF1MetricConfig(MetricConfig):
    __module_path__: ClassVar[str] =  "atria_metrics.core.entity_labeling.layout_f1.layout_f1"
    name: Literal["layout_f1"] = "layout_f1"
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
