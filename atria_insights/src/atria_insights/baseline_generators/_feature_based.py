from __future__ import annotations

import random
from collections import OrderedDict
from pathlib import Path
from typing import Literal

import torch
from atria_logger import get_logger
from atria_registry import ModuleConfig

from atria_insights.baseline_generators._base import BaselineGenerator
from atria_insights.data_types._features import BatchFeatures
from atria_insights.storage.sample_cache_managers._features_cacher import FeaturesCacher

logger = get_logger(__name__)


class FeatureBasedBaselineGeneratorConfig(ModuleConfig):
    module_path: str | None = (
        "atria_insights.baseline_generators._feature_based.FeatureBasedBaselineGenerator"
    )
    type: Literal["feature_based"] = "feature_based"
    features_path: str | None = None
    num_baselines: int = 100
    seed: int = 42


class FeatureBasedBaselineGenerator(
    BaselineGenerator[FeatureBasedBaselineGeneratorConfig]
):
    __config__ = FeatureBasedBaselineGeneratorConfig

    def __init__(
        self, config: FeatureBasedBaselineGeneratorConfig | None = None
    ) -> None:
        super().__init__(config=config)
        assert self.config.features_path is not None, (
            "features_path must be provided in the configuration."
        )
        self._features_cacher = FeaturesCacher(
            cache_dir=Path(self.config.features_path).parent,
            file_name=Path(self.config.features_path).name,
        )
        self._num_baselines = self.config.num_baselines
        self._seed = self.config.seed
        self._features: OrderedDict[str, torch.Tensor] | None = None

    def _get_rand_feature_keys(self) -> list[str]:
        baseline_keys = self._features_cacher.list_sample_keys()

        random.seed(self._seed)
        random.shuffle(baseline_keys)

        return baseline_keys[: self._num_baselines]

    def _load_features(self) -> OrderedDict[str, torch.Tensor]:
        features_batch = []
        for sample_key in self._get_rand_feature_keys():
            features = self._features_cacher.load_sample(sample_key)
            features_batch.append(features)
        batch_features = BatchFeatures.fromlist(features_batch)
        return batch_features.as_ordered_dict()

    def __call__(  # type: ignore[override]
        self, inputs: torch.Tensor | OrderedDict[str, torch.Tensor]
    ) -> OrderedDict[str, torch.Tensor] | torch.Tensor:
        if self._features is None:
            self._features = self._load_features()

        if isinstance(inputs, OrderedDict):
            baselines = OrderedDict()
            for input_key in inputs.keys():
                baselines[input_key] = self._features[input_key].to(
                    inputs[input_key].device
                )
            return baselines
        else:
            assert isinstance(inputs, torch.Tensor), (
                f"Unexpected input type: {type(inputs)}"
            )
            assert len(self._features) == 1, (
                "Expected single feature for tensor input, "
                f"but found {len(self._features)} features."
            )
            first_key = next(iter(self._features))
            return self._features[first_key].to(inputs.device)
