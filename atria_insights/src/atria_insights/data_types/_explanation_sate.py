from __future__ import annotations

from collections import OrderedDict
from typing import Any, Self

import torch
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa: F401
from atria_logger import get_logger
from pydantic import (
    BaseModel,
    ConfigDict,
    field_serializer,
    field_validator,
    model_validator,
)
from torchxai.data_types import (
    ExplanationTarget,
    ExplanationTargetType,
    MultiIndexTargetAcrossBatch,
    MultiIndexTargetPerSample,
    NoTarget,
    SingleTargetAcrossBatch,
    SingleTargetPerSample,
)

BaselineType = torch.Tensor | tuple[torch.Tensor]

logger = get_logger(__name__)


def _extract_sample_target_from_batch(
    target: ExplanationTargetType | list[ExplanationTargetType],
    sample_idx: int,
    batch_size: int,
):
    if isinstance(target, list):
        assert all(isinstance(t, SingleTargetAcrossBatch) for t in target), (
            "In multi-target case, we only support SingleTargetAcrossBatch targets. This means all samples in the batch share the same multi-target list."
        )
        return target
    else:
        if isinstance(target, SingleTargetPerSample):
            assert len(target.indices) == batch_size, (
                "Target list length does not match batch size"
            )
            return SingleTargetPerSample(
                indices=[target.indices[sample_idx]],  # single target for 1 batch
                names=[target.names[sample_idx]],
            )
        elif isinstance(target, MultiIndexTargetPerSample):
            assert len(target.indices) == batch_size, (
                "Target list length does not match batch size"
            )
            return MultiIndexTargetPerSample(
                indices=[target.indices[sample_idx]]  # single target for 1 batch
            )
        else:
            return target


def _create_batch_target_from_sample_target(
    targets: list[ExplanationTargetType | list[ExplanationTargetType]],
    is_multitarget: bool,
) -> ExplanationTargetType | list[ExplanationTargetType]:
    if is_multitarget:
        # if all targets are SingleTargetAcrossBatch, we can combine them into a list
        t0 = targets[0]
        assert all(targets[i] == t0 for i in range(1, len(targets))), (
            "In multi-target case, we only support SingleTargetAcrossBatch targets with the same values. "
            "This means all samples in the batch share the same multi-target list."
        )
        return targets[0]
    else:
        if isinstance(
            targets[0], SingleTargetAcrossBatch | MultiIndexTargetAcrossBatch | NoTarget
        ):
            return targets[0]
        elif isinstance(targets[0], SingleTargetPerSample):
            indices = []
            names = []
            for t in targets:
                assert isinstance(t, SingleTargetPerSample), (
                    "In single-target case, all targets must be of the same type: either SingleTargetPerSample or MultiIndexTargetPerSample."
                )
                indices.append(t.indices[0])  # single target for 1 batch
                names.append(t.names[0])
            return SingleTargetPerSample(indices=indices, names=names)
        elif all(isinstance(t, MultiIndexTargetPerSample) for t in targets):
            indices = []
            for t in targets:
                assert isinstance(t, MultiIndexTargetPerSample), (
                    "In single-target case, all targets must be of the same type: either SingleTargetPerSample or MultiIndexTargetPerSample."
                )
                indices.append(t.indices[0])  # single target for 1 batch
            return MultiIndexTargetPerSample(indices=indices)
        else:
            raise ValueError(
                "In single-target case, all targets must be of the same type: either SingleTargetPerSample or MultiIndexTargetPerSample."
            )


def _to_device(obj, device):
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, OrderedDict):
        return OrderedDict({k: _to_device(v, device) for k, v in obj.items()})
    if isinstance(obj, tuple):
        return tuple(_to_device(v, device) for v in obj)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device(v, device) for v in obj]
    return obj


class BatchExplanationInputs(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )
    sample_id: list[str]
    inputs: tuple[torch.Tensor, ...] | None = None
    additional_forward_args: tuple[Any, ...] | None = None
    baselines: tuple[torch.Tensor, ...] | None = None
    feature_mask: tuple[torch.Tensor, ...] | None = None
    target: ExplanationTargetType | list[ExplanationTargetType] = NoTarget()
    frozen_features: list[torch.Tensor] | None = None
    feature_keys: tuple[str, ...] | None = None

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

        for elem in [self.inputs, self.baselines, self.feature_mask]:
            _validate_batch_size_in_tuple(elem)
            _validate_feature_size_in_tuple(elem, self.feature_keys)

        return self

    @field_validator("target", mode="before")
    @classmethod
    def convert_target(cls, v):
        if (
            isinstance(v, ExplanationTarget)
            or isinstance(v, list)
            and all(isinstance(t, ExplanationTarget) for t in v)
        ):
            return v
        validated = ExplanationTarget.from_raw_input(v)
        return validated

    @field_serializer("target")
    def serialize_target(self, v: ExplanationTarget | list[ExplanationTarget]) -> Any:
        if isinstance(v, list):
            return [t.value for t in v]
        return v.value

    @field_validator("additional_forward_args", mode="before")
    @classmethod
    def to_tuple(cls, v):
        if v is None:
            return None
        if isinstance(v, tuple):
            return v
        if isinstance(v, list):
            return tuple(v)
        return (v,)

    @field_validator("frozen_features", mode="before")
    @classmethod
    def normalize_frozen_features(cls, v):
        if v is None:
            return None
        if isinstance(v, torch.Tensor):
            return [v]
        if isinstance(v, (list, tuple)):
            return list(v)
        return [v]

    # ------------------------------
    # Move everything to device
    # ------------------------------
    def to(self, device: str | torch.device = "cpu") -> Self:
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

    @property
    def model_inputs(self) -> tuple[torch.Tensor, ...]:
        assert isinstance(self.inputs, OrderedDict), (
            f"Expected OrderedDict, got {type(self.inputs)}"
        )
        return tuple(self.inputs.values()) + (
            self.additional_forward_args
            if self.additional_forward_args is not None
            else ()
        )

    def get_explainer_kwargs(self) -> dict[str, Any]:
        explainer_kwargs = {
            "inputs": self.inputs,
            "baselines": self.baselines,
            "feature_mask": self.feature_mask,
            "additional_forward_args": self.additional_forward_args,
            "target": [t.value for t in self.target]
            if isinstance(self.target, list)
            else self.target.value,
            "frozen_features": self.frozen_features,
        }
        return explainer_kwargs


class BatchExplanationState(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )

    # sample
    sample_id: list[str]

    # target
    target: ExplanationTargetType | list[ExplanationTargetType]

    # feature keys
    feature_keys: tuple[str, ...]

    # frozen_features
    frozen_features: list[torch.Tensor] | None = None

    # explanations
    explanations: tuple[torch.Tensor, ...]

    # model_outputs
    model_outputs: torch.Tensor

    # is multitarget
    @property
    def is_multitarget(self) -> bool:
        if isinstance(self.target, list):
            return True
        return False

    def tolist(self) -> list[SampleExplanationState]:
        explanation_states = []
        batch_size = len(self.sample_id)
        for sample_idx in range(batch_size):
            target = _extract_sample_target_from_batch(
                target=self.target, sample_idx=sample_idx, batch_size=batch_size
            )
            sample_expl_state = SampleExplanationState(
                sample_id=self.sample_id[sample_idx],
                target=target,
                feature_keys=self.feature_keys,
                frozen_features=(
                    self.frozen_features[sample_idx]
                    if self.frozen_features is not None
                    else None
                ),
                explanations=tuple(
                    explanation[sample_idx].unsqueeze(0)
                    for explanation in self.explanations
                ),
                model_outputs=self.model_outputs[sample_idx].unsqueeze(0),
                is_multitarget=self.is_multitarget,
            )
            explanation_states.append(sample_expl_state)
        return explanation_states

    @classmethod
    def fromlist(cls, data: list[SampleExplanationState]) -> BatchExplanationState:
        sample_id = [d.sample_id for d in data]
        is_multitarget = data[0].is_multitarget
        assert all(d.is_multitarget == is_multitarget for d in data), (
            "All SampleExplanationState instances must have the same is_multitarget value."
        )
        feature_keys = data[0].feature_keys
        assert all(d.feature_keys == feature_keys for d in data), (
            "All SampleExplanationState instances must have the same feature_keys."
        )

        # collect frozen features
        frozen_features = None
        if data[0].frozen_features is not None:
            frozen_features = []
            for d in data:
                assert d.frozen_features is not None, (
                    "Either all or none of the SampleExplanationState instances must have frozen_features."
                )
                frozen_features.append(d.frozen_features)

        explanations = tuple(
            torch.cat([d.explanations[i] for d in data], dim=0)
            for i in range(len(data[0].explanations))
        )

        model_outputs = torch.cat([d.model_outputs for d in data], dim=0)

        # collect targets
        return cls(
            sample_id=sample_id,
            target=_create_batch_target_from_sample_target(
                targets=[d.target for d in data], is_multitarget=is_multitarget
            ),
            feature_keys=feature_keys,
            frozen_features=frozen_features,
            explanations=explanations,
            model_outputs=model_outputs,
        )


class SampleExplanationState(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )

    # sample
    sample_id: str

    # target
    target: ExplanationTargetType | list[ExplanationTargetType]

    # feature keys
    feature_keys: tuple[str, ...]

    # frozen_features
    frozen_features: torch.Tensor | None = None

    # explanations
    explanations: tuple[torch.Tensor, ...]

    # model_outputs
    model_outputs: torch.Tensor

    # is_multitarget
    is_multitarget: bool = False
