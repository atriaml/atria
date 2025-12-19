from atria_metrics.core import METRIC, MetricConfig


@METRIC.register("coco_eval")
class CocoEvalMetricConfig(MetricConfig):
    name: str = "coco_eval"
    module_path: str | None = "atria_metrics.core.detection.cocoeval.CocoEvalMetric"
