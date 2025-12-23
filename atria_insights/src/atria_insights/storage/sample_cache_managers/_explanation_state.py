from pathlib import Path

import torch
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa: F401
from atria_logger import get_logger

from atria_insights.data_types._explanation_state import SampleExplanationState
from atria_insights.model_pipelines._common import ExplainableModelPipelineConfig
from atria_insights.storage.data_cachers._common import CacheData
from atria_insights.storage.sample_cache_managers._base import BaseSampleCacheManager
from atria_insights.utilities._common import (
    _map_tensor_dicts_to_tuples,
    _map_tensor_tuples_to_keys,
)

logger = get_logger(__name__)


class ExplanationStateCacher(BaseSampleCacheManager[SampleExplanationState]):
    def __init__(self, cache_dir: str | Path, config: ExplainableModelPipelineConfig):
        # create a child cache dir for the given explainer
        super().__init__(
            cache_dir=Path(cache_dir) / config.explainer.type,
            file_name=f"explanations-{config.hash}.hdf5",
        )
        self._config = config
        self._dump_config()

    def _dump_config(self) -> dict:
        with open("config.yaml", "w") as f:
            f.write(self._config.to_yaml())
            return self._config.model_dump()

    def _to_cache_data(self, data: SampleExplanationState) -> CacheData:
        if data.target is not None:
            target = (
                [t.model_dump() for t in data.target]
                if isinstance(data.target, list)
                else data.target.model_dump()
            )
        else:
            target = None

        # we store explaantions list for multi-target scenario as stacked tensors
        if isinstance(data.explanations, list):
            per_target_explanations = data.explanations
            per_target_explanations = tuple(
                map(list, zip(*per_target_explanations, strict=True))
            )

            # convert lists back to tensors
            explanations = tuple(
                torch.stack(tensors, dim=0) for tensors in per_target_explanations
            )

            # map to dict These are per target (batch_size, T, ...)
            explanations = _map_tensor_tuples_to_keys(explanations, data.feature_keys)
        else:
            explanations = _map_tensor_tuples_to_keys(
                data.explanations, data.feature_keys
            )

        return CacheData(
            sample_id=data.sample_id,
            attrs={
                "sample_id": data.sample_id,
                "target": target,
                "feature_keys": data.feature_keys,
                "frozen_features": data.frozen_features,
                "config_hash": self._config.hash,
                "is_multitarget": data.is_multitarget,
            },
            tensors={"model_outputs": data.model_outputs, "explanations": explanations},
        )

    def _from_cache_data(self, data: CacheData) -> SampleExplanationState:
        assert data.attrs is not None, "attrs must be provided in CacheData."
        assert data.tensors is not None, "tensors must be provided in CacheData."

        feature_keys = data.attrs.get("feature_keys")
        assert feature_keys is not None, "feature_keys must be provided in attrs."

        if data.attrs.get("is_multitarget", False):
            explanations = _map_tensor_dicts_to_tuples(
                data.tensors["explanations"], tuple(feature_keys)
            )

            # this will be tuple of (batch_size, T, ...) -> map back to list of tuples of (batch_size, ...)
            explanations = [
                tuple(explanations[tgt_idx][i] for tgt_idx in range(len(explanations)))
                for i in range(explanations[0].shape[0])
            ]
        else:
            explanations = _map_tensor_dicts_to_tuples(
                data.tensors["explanations"], tuple(feature_keys)
            )
        model_outputs = data.tensors["model_outputs"]
        assert isinstance(model_outputs, torch.Tensor), (
            "model_outputs must be a torch.Tensor."
        )
        return SampleExplanationState(
            sample_id=data.attrs["sample_id"],
            target=data.attrs["target"],
            feature_keys=feature_keys,
            frozen_features=data.attrs.get("frozen_features"),
            model_outputs=model_outputs,
            explanations=explanations,
        )
