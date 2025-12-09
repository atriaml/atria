from atria_metrics import METRIC, MetricConfig


@METRIC.register("coco_eval")
class CocoEvalMetricConfig(MetricConfig):
    module_path: str | None = "atria_metrics.impl.detection.cocoeval.COCOEvalMetric"
    output_transform: Callable = _cocoeval_output_transform
