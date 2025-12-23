from pathlib import Path

import torch
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa: F401
from atria_logger import get_logger
from torchxai.data_types import ExplanationTargetType

from atria_insights.data_types._explanation_sate import SampleExplanationState
from atria_insights.model_pipelines._common import ExplainableModelPipelineConfig
from atria_insights.storage.data_cachers._common import CacheData
from atria_insights.storage.sample_cache_managers._base import BaseSampleCacheManager

logger = get_logger(__name__)


def _map_tensor_tuples_to_keys(tensor_tuple: tuple[torch.Tensor, ...], keys):
    return dict(zip(keys, tensor_tuple, strict=True))


def _serialize_target(
    target: ExplanationTargetType | list[ExplanationTargetType],
) -> dict | list[dict]:
    if isinstance(target, list):
        return [t.model_dump() for t in target]
    else:
        return target.model_dump()


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
        target = _serialize_target(data.target)
        return CacheData(
            sample_id=data.sample_id,
            attrs={
                "sample_id": data.sample_id,
                "target": target,
                "feature_keys": data.feature_keys,
                "frozen_features": data.frozen_features,
                "config_hash": self._config.hash,
            },
            tensors={
                "model_outputs": data.model_outputs,
                "explanations": _map_tensor_tuples_to_keys(
                    data.explanations, data.feature_keys
                ),
            },
        )

    def _from_cache_data(self, data: CacheData) -> SampleExplanationState:
        assert data.attrs is not None, "attrs must be provided in CacheData."
        assert data.tensors is not None, "tensors must be provided in CacheData."

        feature_keys = data.attrs.get("feature_keys")
        assert feature_keys is not None, "feature_keys must be provided in attrs."

        # map back arrays to feature keys
        def _map_tensor_dicts_to_tuples(
            tensor_tuple: dict[str, torch.Tensor] | torch.Tensor, keys: tuple[str]
        ) -> tuple[torch.Tensor, ...]:
            if isinstance(tensor_tuple, torch.Tensor):
                return (tensor_tuple,)
            return tuple(tensor_tuple[feature_key] for feature_key in keys)

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
