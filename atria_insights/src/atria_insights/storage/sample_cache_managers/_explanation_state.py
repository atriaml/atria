import json
from pathlib import Path

import torch
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa: F401
from atria_logger import get_logger
from torchxai.data_types import SampleExplanationTarget

from atria_insights.data_types._explanation_state import (
    MultiTargetSampleExplanation,
    SampleExplanation,
    SampleExplanationState,
)
from atria_insights.model_pipelines._common import ExplainableModelPipelineConfig
from atria_insights.storage.data_cachers._common import SerializableSampleData
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

    def _serialize_type(self, data: SampleExplanationState) -> SerializableSampleData:
        if data.target is not None:
            target = (
                [t.model_dump() for t in data.target]
                if isinstance(data.target, list)
                else data.target.model_dump()
            )
        else:
            target = None

        # we store explaantions list for multi-target scenario as stacked tensors
        if isinstance(data.explanations, MultiTargetSampleExplanation):
            assert data.explanations.n_features == len(data.feature_keys), (
                "Number of explanation tensors must match number of feature keys."
            )

            per_target_sample_explanations = data.explanations.value

            # map to dict These are per target (batch_size, T, ...)
            # this is a list of dicts of feature_key -> tensor
            explanations = [
                _map_tensor_tuples_to_keys(sample_explanations.value, data.feature_keys)
                for sample_explanations in per_target_sample_explanations
            ]

            # we need to stack per target explanations to get (T, ...)
            # so we have a dict of feature_key -> tensor of shape (T, ...)
            explanations = {
                key: torch.stack(
                    [explanations[tgt_idx][key] for tgt_idx in range(len(explanations))]
                )
                for key in data.feature_keys
            }
        else:
            assert data.explanations.n_features == len(data.feature_keys), (
                "Number of explanation tensors must match number of feature keys."
            )
            explanations = _map_tensor_tuples_to_keys(
                data.explanations.value, data.feature_keys
            )

        return SerializableSampleData(
            sample_id=data.sample_id,
            attrs={
                "sample_id": data.sample_id,
                "target": json.dumps(target),
                "feature_keys": list(data.feature_keys),
                "frozen_features": data.frozen_features.tolist()
                if data.frozen_features is not None
                else None,
                "config_hash": self._config.hash,
                "is_multitarget": data.is_multitarget,
            },
            tensors={"model_outputs": data.model_outputs, "explanations": explanations},
        )

    def _deserialize_type(self, data: SerializableSampleData) -> SampleExplanationState:
        assert data.attrs is not None, "attrs must be provided in CacheData."
        assert data.tensors is not None, "tensors must be provided in CacheData."
        sample_id = data.attrs.get("sample_id")
        assert isinstance(sample_id, str), "sample_id must be a string."

        feature_keys = data.attrs.get("feature_keys")
        assert isinstance(feature_keys, list), "feature_keys must be a list."
        feature_keys = [str(fk) for fk in feature_keys]

        target = data.attrs.get("target")
        assert isinstance(target, str), "target must be a string."
        target = json.loads(target)

        is_multitarget = data.attrs.get("is_multitarget", False)

        frozen_features = data.attrs.get("frozen_features")
        if frozen_features is not None:
            assert isinstance(frozen_features, list), (
                "frozen_features must be a list if provided."
            )
            frozen_features = torch.Tensor(frozen_features)

        if is_multitarget:
            explanations = _map_tensor_dicts_to_tuples(
                data.tensors["explanations"], tuple(feature_keys)
            )

            # this will be tuple of (batch_size, T, ...) -> map back to list of tuples of (batch_size, ...)
            explanations = [
                tuple(explanations[tgt_idx][i] for tgt_idx in range(len(explanations)))
                for i in range(explanations[0].shape[0])
            ]

            # map to MultiTargetSampleExplanation
            explanations = MultiTargetSampleExplanation(
                value=[
                    SampleExplanation(value=explanations[i])
                    for i in range(len(explanations))
                ]
            )
        else:
            assert isinstance(data.tensors["explanations"], dict), (
                "explanations must be a dict for single-target scenario."
            )
            explanations = SampleExplanation(
                value=_map_tensor_dicts_to_tuples(
                    data.tensors["explanations"], tuple(feature_keys)
                )
            )

        if is_multitarget:
            assert isinstance(target, list), (
                "target must be a list for multi-target scenario."
            )
            target = [SampleExplanationTarget.model_validate(t) for t in target]
        else:
            target = (
                SampleExplanationTarget.model_validate(target)
                if target is not None
                else None
            )
        model_outputs = data.tensors["model_outputs"]
        assert isinstance(model_outputs, torch.Tensor), (
            "model_outputs must be a torch.Tensor."
        )
        return SampleExplanationState(
            sample_id=sample_id,
            target=target,
            feature_keys=tuple(feature_keys),
            frozen_features=frozen_features,
            model_outputs=model_outputs,
            explanations=explanations,
        )
