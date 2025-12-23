from pathlib import Path

import torch
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa: F401
from atria_logger import get_logger

from atria_insights.data_types._metric_data import SampleMetricData
from atria_insights.explainability_metrics._base import ExplainabilityMetricConfig
from atria_insights.storage.data_cachers._common import CacheData
from atria_insights.storage.sample_cache_managers._base import BaseSampleCacheManager

logger = get_logger(__name__)


class MetricDataCacher(BaseSampleCacheManager[SampleMetricData]):
    def __init__(self, cache_dir: str | Path, config: ExplainabilityMetricConfig):
        # create a child cache dir for the given explainer

        super().__init__(
            cache_dir=Path(cache_dir), file_name=f"{config.type}-{config.hash}.hdf5"
        )
        self._config = config

    def _to_cache_data(self, data: SampleMetricData) -> CacheData:
        # find all tensors in data
        tensors = {}
        attrs = {}
        for key, value in data.data.items():
            if isinstance(value, torch.Tensor):
                tensors[key] = value
            else:
                attrs[key] = value

        return CacheData(
            sample_id=data.sample_id,
            attrs={
                "sample_id": data.sample_id,
                "config_hash": self._config.hash,
                **attrs,
            },
            tensors=tensors,
        )

    def _from_cache_data(self, data: CacheData) -> SampleMetricData:
        assert data.attrs is not None, "attrs must be provided in CacheData."
        assert data.tensors is not None, "tensors must be provided in CacheData."

        return SampleMetricData(
            sample_id=data.attrs["sample_id"], data={**data.attrs, **data.tensors}
        )
