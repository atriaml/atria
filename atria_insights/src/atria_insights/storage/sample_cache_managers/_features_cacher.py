from pathlib import Path

from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa: F401
from atria_logger import get_logger

from atria_insights.data_types._features import SampleFeatures
from atria_insights.storage.data_cachers._common import CacheData
from atria_insights.storage.sample_cache_managers._base import BaseSampleCacheManager
from atria_insights.utilities._common import (
    _map_tensor_dicts_to_tuples,
    _map_tensor_tuples_to_keys,
)

logger = get_logger(__name__)


class FeaturesCacher(BaseSampleCacheManager[SampleFeatures]):
    def __init__(self, cache_dir: str | Path, file_name: str = "features.hdf5"):
        # create a child cache dir for the given explainer
        super().__init__(cache_dir=Path(cache_dir), file_name=file_name)

    def _to_cache_data(self, data: SampleFeatures) -> CacheData:
        return CacheData(
            sample_id=data.sample_id,
            attrs={"sample_id": data.sample_id, "feature_keys": data.feature_keys},
            tensors={
                "features": _map_tensor_tuples_to_keys(data.features, data.feature_keys)
            },
        )

    def _from_cache_data(self, data: CacheData) -> SampleFeatures:
        assert data.attrs is not None, "attrs must be provided in CacheData."
        assert data.tensors is not None, "tensors must be provided in CacheData."

        feature_keys = data.attrs.get("feature_keys")
        assert feature_keys is not None, "feature_keys must be provided in attrs."

        features = _map_tensor_dicts_to_tuples(
            data.tensors["features"], tuple(feature_keys)
        )
        return SampleFeatures(
            sample_id=data.attrs["sample_id"],
            feature_keys=feature_keys,
            features=features,
        )
