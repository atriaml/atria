from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from atria_models.core.types.model_outputs import ClassificationModelOutput

if TYPE_CHECKING:
    from ignite.metrics import Metric


def f1_score(output_transform: Callable, device: str = "cpu") -> Metric:
    from ignite.metrics import Precision, Recall

    # use ignite arthematics of metrics to compute f1 score
    # unnecessary complication
    precision = Precision(
        average=False, output_transform=output_transform, device=device
    )
    recall = Recall(average=False, output_transform=output_transform, device=device)
    return (precision * recall * 2 / (precision + recall)).mean()


def _output_transform(output: ClassificationModelOutput):
    assert isinstance(output, ClassificationModelOutput), (
        f"Expected {ClassificationModelOutput}, got {type(output)}"
    )
    return output.logits, output.gt_label_value
