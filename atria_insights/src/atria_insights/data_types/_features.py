from __future__ import annotations

from collections import OrderedDict

import torch
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa: F401
from atria_logger import get_logger
from pydantic import BaseModel, ConfigDict, model_validator

logger = get_logger(__name__)


class BatchFeatures(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    sample_id: list[str]
    feature_keys: tuple[str, ...]
    features: tuple[torch.Tensor, ...]

    def tolist(self) -> list[SampleFeatures]:
        for feature_key, feature in zip(self.feature_keys, self.features, strict=True):
            print(f"{feature_key}: {feature.shape}")
        return [
            SampleFeatures(
                sample_id=sid,
                feature_keys=self.feature_keys,
                features=tuple(feature[i] for feature in self.features),
            )
            for i, sid in enumerate(self.sample_id)
        ]

    @classmethod
    def fromlist(cls, data: list[SampleFeatures]) -> BatchFeatures:
        if not data:
            raise ValueError("data list is empty")

        # make sure all samples have the same feature keys
        first_feature_keys = data[0].feature_keys
        for d in data:
            if d.feature_keys != first_feature_keys:
                raise ValueError(
                    "All SampleFeatures must have the same feature_keys to form a batch."
                )

        feature_keys = data[0].feature_keys
        sample_ids = [d.sample_id for d in data]
        features = tuple(
            torch.stack([d.features[i] for d in data], dim=0)
            for i in range(len(feature_keys))
        )

        return cls(sample_id=sample_ids, feature_keys=feature_keys, features=features)

    def as_ordered_dict(self) -> OrderedDict:
        return OrderedDict(zip(self.feature_keys, self.features, strict=True))


class SampleFeatures(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    sample_id: str
    feature_keys: tuple[str, ...]
    features: tuple[torch.Tensor, ...]

    @model_validator(mode="after")
    def _validate_features_length(self) -> SampleFeatures:
        assert len(self.features) > 0, "features must contain at least one tensor."
        if len(self.feature_keys) != len(self.features):
            raise ValueError(
                "Length of feature_keys must match length of features tensors."
            )
        return self
