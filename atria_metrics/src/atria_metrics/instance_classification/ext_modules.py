from atria_metrics.instance_classification.output_transforms import _output_transform
from atria_metrics.registry import METRIC

for metric_name, module, kwargs in [
    ("accuracy", "ignite.metrics.Accuracy", {"is_multilabel": False}),
    ("precision", "ignite.metrics.Precision", {"average": True}),
    ("recall", "ignite.metrics.Recall", {"average": True}),
    ("confusion_matrix", "ignite.metrics.ConfusionMatrix", {"average": "recall"}),
]:
    METRIC.register(
        metric_name, output_transform=_output_transform, device="cpu", **kwargs
    )(module)
