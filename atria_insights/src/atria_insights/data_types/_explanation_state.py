from __future__ import annotations

from typing import Self

import torch
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa: F401
from atria_logger import get_logger
from pydantic import BaseModel, ConfigDict
from torchxai.data_types import BatchExplanationTarget, SampleExplanationTarget

from atria_insights.utilities._common import _to_device

BaselineType = torch.Tensor | tuple[torch.Tensor]

logger = get_logger(__name__)


class SampleExplanationState(BaseModel):
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
    explanations: tuple[torch.Tensor, ...] | list[tuple[torch.Tensor, ...]]
    model_outputs: torch.Tensor
    is_multitarget: bool = False


class BatchExplanationState(BaseModel):
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
    explanations: tuple[torch.Tensor, ...] | list[tuple[torch.Tensor, ...]]
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

        for sample_idx in range(batch_size):
            if isinstance(self.explanations, list):
                explanation = [
                    tuple(
                        explanation[sample_idx].unsqueeze(0)
                        for explanation in self.explanations
                    )
                ]
            else:
                explanation = tuple(
                    explanation[sample_idx].unsqueeze(0)
                    for explanation in self.explanations
                )

            sample_expl_state = SampleExplanationState(
                sample_id=self.sample_id[sample_idx],
                target=targets[sample_idx],
                feature_keys=self.feature_keys,
                frozen_features=(
                    self.frozen_features[sample_idx]
                    if self.frozen_features is not None
                    else None
                ),
                explanations=explanation,
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

        if is_multitarget:
            explanations_list = [d.explanations for d in data]
            # explanations_list is now a list of lists of tuples
            # we need to transpose this into list of tuples of lists
            # so that we can batch per target
            transposed_explanations = list(
                map(list, zip(*explanations_list, strict=True))
            )
            explanations = [
                tuple(
                    torch.cat(
                        [
                            d_explanations[tgt_idx]
                            for d_explanations in explanations_list
                        ],
                        dim=0,
                    )
                    for tgt_idx in range(len(transposed_explanations))
                )
                for tgt_idx in range(len(transposed_explanations))
            ]
        else:
            explanations = tuple(
                torch.cat([d.explanations[i] for d in data], dim=0)
                for i in range(len(data[0].explanations))
            )

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
                "explanations": _to_device(self.explanations, device),
                "frozen_features": _to_device(self.frozen_features, device),
            }
        )
