from __future__ import annotations

from typing import Self

import torch
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa: F401
from atria_logger import get_logger
from atria_types._utilities._repr import RepresentationMixin
from pydantic import BaseModel, ConfigDict, model_validator

from atria_insights.data_types._targets import (
    BatchExplanationTarget,
    SampleExplanationTarget,
)
from atria_insights.utilities._common import _to_device

BaselineType = torch.Tensor | tuple[torch.Tensor]

logger = get_logger(__name__)


class SampleExplanation(RepresentationMixin, BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )
    value: tuple[torch.Tensor, ...]

    @property
    def device(self) -> torch.device:
        return self.value[0].device

    @property
    def batch_size(self) -> int:
        return self.value[0].shape[0]

    @property
    def n_features(self) -> int:
        return len(self.value)

    @model_validator(mode="after")
    def check_batch_size(self) -> Self:
        batch_sizes = {tensor.shape[0] for tensor in self.value}
        if len(batch_sizes) != 1:
            raise ValueError(
                f"All tensors in explanation must have the same batch size, got batch sizes: {batch_sizes}"
            )
        return self


class MultiTargetSampleExplanation(RepresentationMixin, BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )
    value: list[SampleExplanation]

    @property
    def device(self) -> torch.device:
        return self.value[0].device

    @property
    def batch_size(self) -> int:
        return self.value[0].batch_size

    @property
    def n_targets(self) -> int:
        return len(self.value)

    @property
    def n_features(self) -> int:
        return self.value[0].n_features


class BatchExplanation(RepresentationMixin, BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )
    value: tuple[torch.Tensor, ...]

    @property
    def device(self) -> torch.device:
        return self.value[0].device

    @property
    def batch_size(self) -> int:
        return self.value[0].shape[0]

    @property
    def n_features(self) -> int:
        return len(self.value)

    @classmethod
    def fromlist(cls, data: list[SampleExplanation]) -> BatchExplanation:
        explanations = tuple(
            torch.cat([sample_expl.value[i] for sample_expl in data], dim=0)
            for i in range(len(data[0].value))
        )
        return cls(value=explanations)

    def tolist(self) -> list[SampleExplanation]:
        batch_size = self.batch_size
        sample_explanations = []
        for sample_idx in range(batch_size):
            sample_tensors = tuple(
                tensor[sample_idx].unsqueeze(0) for tensor in self.value
            )
            sample_explanations.append(SampleExplanation(value=sample_tensors))
        return sample_explanations

    def to_device(self, device: str | torch.device = "cpu") -> Self:
        return self.model_copy(
            update={"value": tuple(tensor.to(device) for tensor in self.value)}
        )


class MultiTargetBatchExplanation(RepresentationMixin, BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )
    value: list[BatchExplanation]

    @property
    def device(self) -> torch.device:
        return self.value[0].device

    @property
    def batch_size(self) -> int:
        return self.value[0].batch_size

    @property
    def n_targets(self) -> int:
        return len(self.value)

    @property
    def n_features(self) -> int:
        return self.value[0].n_features

    @classmethod
    def fromlist(
        cls, data: list[MultiTargetSampleExplanation]
    ) -> MultiTargetBatchExplanation:
        # make sure all targets are same
        first_n_target = data[0].n_targets
        assert all(d.n_targets == first_n_target for d in data), (
            "All SampleMultiTargetExplanation instances must have the same number of targets."
        )

        # make sure all features are same
        first_n_features = data[0].n_features
        assert all(d.n_features == first_n_features for d in data), (
            "All SampleMultiTargetExplanation instances must have the same number of features."
        )

        # for target explanations we need to collect per target
        per_target_batch_explanations = []
        for target_idx in range(first_n_target):
            batch_explanations = BatchExplanation.fromlist(
                [d.value[target_idx] for d in data]
            )
            per_target_batch_explanations.append(batch_explanations)
        batched = cls(value=per_target_batch_explanations)

        assert batched.batch_size == len(data), (
            "Batched explanation batch size does not match number of samples."
        )
        assert batched.n_targets == data[0].n_targets, (
            "Number of targets in batched explanation does not match number of targets in samples."
        )
        assert batched.n_features == data[0].n_features, (
            "Number of features in batched explanation does not match number of features in samples."
        )
        return batched

    def tolist(self) -> list[MultiTargetSampleExplanation]:
        per_target_sample_explanations = [
            target_batch_expl.tolist() for target_batch_expl in self.value
        ]
        # now we need to transpose per target list into per sample list
        n_samples = self.batch_size
        multi_target_sample_explanations = []
        for sample_idx in range(n_samples):
            sample_explanations = [
                per_target_sample_explanations[target_idx][sample_idx]
                for target_idx in range(self.n_targets)
            ]
            multi_target_sample_explanations.append(
                MultiTargetSampleExplanation(value=sample_explanations)
            )
        return multi_target_sample_explanations

    def to_device(self, device: str | torch.device = "cpu") -> Self:
        return self.model_copy(
            update={"value": [be.to_device(device) for be in self.value]}
        )


class SampleExplanationState(RepresentationMixin, BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )

    sample_id: str
    target: SampleExplanationTarget | list[SampleExplanationTarget] | None = None
    feature_keys: tuple[str, ...]
    frozen_features: torch.Tensor | None = None
    explanations: SampleExplanation | MultiTargetSampleExplanation
    model_outputs: torch.Tensor

    @property
    def is_multitarget(self) -> bool:
        if isinstance(self.explanations, MultiTargetSampleExplanation):
            return True
        return False


class BatchExplanationState(RepresentationMixin, BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )

    sample_id: list[str]
    target: BatchExplanationTarget | list[BatchExplanationTarget] | None = None
    feature_keys: tuple[str, ...]
    frozen_features: list[torch.Tensor] | None = None
    explanations: BatchExplanation | MultiTargetBatchExplanation
    model_outputs: torch.Tensor

    @property
    def is_multitarget(self) -> bool:
        if isinstance(self.target, list):
            return True
        return False

    @property
    def batch_size(self) -> int:
        """Return the batch size of the explanations."""
        return len(self.sample_id)

    def tolist(self) -> list[SampleExplanationState]:
        explanation_states = []
        batch_size = len(self.sample_id)

        if self.target is None:
            targets = [None] * batch_size
        elif isinstance(self.target, list):
            total_targets = len(self.target)
            targets = [t.tolist() for t in self.target]
            # in multi target case the targets at this point will contain the following
            # [
            #   # target 1: [SampleExplanationTarget1_sample1, SampleExplanationTarget1_sample2, ...],
            #   # target 2: [SampleExplanationTarget2_sample1, SampleExplanationTarget2_sample2, ...],
            # ]
            # we need to transpose this into
            # [
            #   # sample 1: [SampleExplanationTarget1, SampleExplanationTarget2, SampleExplanationTarget3, ...],
            #   # sample 1: [SampleExplanationTarget1, SampleExplanationTarget2, SampleExplanationTarget3, ...],
            # ]
            targets = list(map(list, zip(*targets, strict=True)))
            assert len(targets) == batch_size, (
                "Length of targets must match batch size."
            )
            assert all(len(tgt_list) == total_targets for tgt_list in targets), (
                "Each sample's target list must match the total number of targets."
            )
            for target in targets:
                assert all(isinstance(t, SampleExplanationTarget) for t in target), (
                    "All targets must be instances of SampleExplanationTarget."
                )
        else:
            targets = self.target.tolist()
            assert len(targets) == batch_size, (
                "Length of targets must match batch size."
            )
            for t in targets:
                assert isinstance(t, SampleExplanationTarget), (
                    "All targets must be instances of SampleExplanationTarget."
                )

        # create explanation states per sample
        explanations = self.explanations.tolist()

        for sample_idx in range(batch_size):
            sample_expl_state = SampleExplanationState(
                sample_id=self.sample_id[sample_idx],
                target=targets[sample_idx],
                feature_keys=self.feature_keys,
                frozen_features=(
                    self.frozen_features[sample_idx]
                    if self.frozen_features is not None
                    else None
                ),
                explanations=explanations[sample_idx],
                model_outputs=self.model_outputs[sample_idx].unsqueeze(0),
            )
            explanation_states.append(sample_expl_state)
        return explanation_states

    @classmethod
    def fromlist(cls, data: list[SampleExplanationState]) -> BatchExplanationState:
        assert len(data) > 0, "Data list must not be empty."

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

        if is_multitarget:
            sample_mt_explanations = []
            for d in data:
                assert isinstance(d.explanations, MultiTargetSampleExplanation), (
                    "All explanations must be of type SampleMultiTargetExplanation in multi-target case."
                )
                sample_mt_explanations.append(d.explanations)
            explanations = MultiTargetBatchExplanation.fromlist(sample_mt_explanations)
        else:
            explanations_list = []
            for d in data:
                assert isinstance(d.explanations, SampleExplanation), (
                    "All explanations must be of type SampleExplanation in single-target case."
                )
                explanations_list.append(d.explanations)
            explanations = BatchExplanation.fromlist(explanations_list)

        model_outputs = torch.cat([d.model_outputs for d in data], dim=0)

        # if target is list we collect lists
        batch_size = len(data)
        target = [d.target for d in data]
        if data[0].target is None:
            target = None
        elif isinstance(target[0], list):
            # in multi target case the data will contain the following
            # [
            #   # sample 1: [SampleExplanationTarget1, SampleExplanationTarget2, SampleExplanationTarget3, ...],
            #   # sample 1: [SampleExplanationTarget1, SampleExplanationTarget2, SampleExplanationTarget3, ...],
            # ]
            # we need to transpose this into
            # [
            #   # target 1: [SampleExplanationTarget1_sample1, SampleExplanationTarget1_sample2, ...],
            #   # target 2: [SampleExplanationTarget2_sample1, SampleExplanationTarget2_sample2, ...],
            # ]
            transposed_targets = list(map(list, zip(*target, strict=True)))

            # now each target can be batched separately
            target = [
                BatchExplanationTarget.fromlist(tgt_list)
                for tgt_list in transposed_targets
            ]
            # verify shapes
            for t in target:
                assert len(t.value) == batch_size, (
                    "Length of batched target must match number of samples."
                )
        else:
            # in case of single target we can directly batch as each taget will be an instance of SampleExplanationTarget
            target = BatchExplanationTarget.fromlist(target)  # type: ignore

            # verify shapes
            assert len(target.value) == batch_size, (
                "Length of batched target must match number of samples."
            )

        # collect targets
        return cls(
            sample_id=sample_id,
            target=target,
            feature_keys=feature_keys,
            frozen_features=frozen_features,
            explanations=explanations,
            model_outputs=model_outputs,
        )

    def to_device(self, device: str | torch.device = "cpu") -> Self:
        return self.model_copy(
            update={
                "explanations": self.explanations.to_device(device),
                "frozen_features": _to_device(self.frozen_features, device),
            }
        )
