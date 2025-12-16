from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from atria_datasets.core.utilities import _resolve_module_from_path
from atria_models.core.types.model_outputs import ClassificationModelOutput
from ignite.metrics import Metric

from atria_metrics.core import MetricConfig

if TYPE_CHECKING:
    import torch
    from ignite.metrics import Metric


def f1_score(output_transform: Callable, device: str | torch.device = "cpu") -> Metric:
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


class ClassificationMetricConfig(MetricConfig):
    def build(  # type: ignore[return]
        self,
        device: torch.device | str,
        stage: Literal["train", "test", "validation"],
        num_classes: int | None = None,
    ) -> Metric:
        assert self.module_path is not None, (
            "module_path must be set to build the module."
        )
        module = _resolve_module_from_path(self.module_path)
        if isinstance(module, type | Callable):
            possible_args = inspect.signature(module.__init__).parameters
            kwargs = {
                arg: value
                for arg, value in {
                    "device": device,
                    "num_classes": num_classes,
                    "stage": stage,
                }.items()
                if arg in possible_args and value is not None
            }

            current_kwargs = self.kwargs
            current_kwargs.update(kwargs)
            return module(**current_kwargs, output_transform=_output_transform)
        else:
            raise TypeError(
                f"Module at path {self.module_path} is neither a class nor a callable."
            )
