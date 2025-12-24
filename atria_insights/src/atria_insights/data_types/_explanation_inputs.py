from __future__ import annotations

from typing import Any, Self

import torch
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa: F401
from atria_logger import get_logger
from pydantic import BaseModel, ConfigDict, model_validator
from torchxai.data_types import BatchExplanationTarget

from atria_insights.utilities._common import _to_device

BaselineType = torch.Tensor | tuple[torch.Tensor]

logger = get_logger(__name__)


class BatchExplanationInputs(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )
    sample_id: list[str]
    inputs: tuple[torch.Tensor, ...]
    additional_forward_args: tuple[Any, ...] | None = None
    baselines: tuple[torch.Tensor, ...] | None = None
    feature_mask: tuple[torch.Tensor, ...] | None = None
    target: BatchExplanationTarget | list[BatchExplanationTarget] | None = None
    frozen_features: list[torch.Tensor] | None = None
    feature_keys: tuple[str, ...]

    @property
    def is_multi_target(self) -> bool:
        if isinstance(self.target, list):
            return True
        return False

    @property
    def batch_size(self) -> int:
        """Return the batch size of the inputs."""
        if self.inputs is not None:
            assert len(self.sample_id) == self.inputs[0].shape[0], (
                "Length of sample_id must match batch size of inputs."
            )
        return len(self.sample_id)

    @model_validator(mode="after")
    def validate_multi_feature_inputs(self) -> Self:
        def _validate_batch_size_in_tuple(t: tuple[torch.Tensor, ...] | None):
            if t is None:
                return
            for i, tensor in enumerate(t):
                assert tensor.shape[0] == self.batch_size, (
                    f"All tensors must have the same batch size. Tensor {i} has batch size {tensor.shape[0]}, expected {self.batch_size}."
                )

        def _validate_feature_size_in_tuple(
            t: tuple[torch.Tensor, ...] | None, feature_keys: tuple[str, ...] | None
        ):
            if t is None:
                return
            if feature_keys is None:
                raise ValueError("feature_keys must be provided.")
            if len(t) != len(feature_keys):
                raise ValueError("Length of feature_keys must match number of tensors.")

        for elem in [
            self.inputs,
            self.feature_mask,
        ]:  # baselines can be of different batch size so we don't validate it here
            _validate_batch_size_in_tuple(elem)
            _validate_feature_size_in_tuple(elem, self.feature_keys)

        return self

    def to_device(self, device: str | torch.device = "cpu") -> Self:
        return self.model_copy(
            update={
                "baselines": _to_device(self.baselines, device),
                "feature_mask": _to_device(self.feature_mask, device),
                "inputs": _to_device(self.inputs, device),
                "additional_forward_args": _to_device(
                    self.additional_forward_args, device
                ),
                "target": _to_device(self.target, device),
                "frozen_features": _to_device(self.frozen_features, device),
            }
        )

    def get_explainer_kwargs(self) -> dict[str, Any]:
        if self.target is None:
            target = None
        else:
            target = (
                [t.value for t in self.target]
                if isinstance(self.target, list)
                else self.target.value,
            )
        explainer_kwargs = {
            "inputs": self.inputs,
            "baselines": self.baselines,
            "feature_mask": self.feature_mask,
            "additional_forward_args": self.additional_forward_args,
            "target": target,
            "frozen_features": self.frozen_features,
        }
        return explainer_kwargs
