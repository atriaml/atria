from collections.abc import Callable

from atria_metrics.core import METRIC, MetricConfig
from atria_metrics.core.detection.cocoeval import _cocoeval_output_transform


@METRIC.register("coco_eval")
class CocoEvalMetricConfig(MetricConfig):
    name: str = "coco_eval"
    module_path: str | None = "atria_metrics.core.detection.cocoeval.CocoEvalMetric"
    output_transform: Callable = _cocoeval_output_transform
