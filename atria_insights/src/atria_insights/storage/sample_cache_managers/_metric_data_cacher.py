from pathlib import Path

import torch
from atria_logger import get_logger

from atria_insights.data_types._metric_data import SampleMetricData
from atria_insights.storage.data_cachers._base import DataCacher
from atria_insights.storage.data_cachers._common import SerializableSampleData
from atria_insights.storage.data_cachers._hdf5 import HDF5DataCacher
from atria_insights.storage.sample_cache_managers._base import BaseSampleCacheManager
from atria_insights.storage.sample_cache_managers._utilities import to_serializable

logger = get_logger(__name__)


class MetricDataCacher(BaseSampleCacheManager[SampleMetricData]):
    def __init__(self, cacher: DataCacher):
        super().__init__(cacher=cacher)

    def _serialize_type(self, data: SampleMetricData) -> SerializableSampleData:
        tensors = {}
        attrs = {}
        for key, value in data.data.items():
            if isinstance(value, torch.Tensor):
                tensors[key] = value
            else:
                attrs[key] = to_serializable(value)

        if data.config is not None:
            attrs["config"] = data.config

        return SerializableSampleData(
            sample_id=data.sample_id,
            attrs={"sample_id": data.sample_id, **attrs},
            tensors=tensors,
        )

    def _deserialize_type(self, data: SerializableSampleData) -> SampleMetricData:
        assert data.attrs is not None, "attrs must be provided in CacheData."

        attrs = dict(data.attrs)
        sample_id = attrs.pop("sample_id", None)
        assert isinstance(sample_id, str), "sample_id must be a string."
        attrs.pop("config_hash", None)
        config = attrs.pop("config", None)

        tensors = data.tensors or {}
        return SampleMetricData(
            sample_id=sample_id,
            data={**attrs, **tensors},  # type: ignore
            config=config,
        )


class H5MetricDataCacher(MetricDataCacher):
    def __init__(self, cache_dir: str):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(
            cacher=HDF5DataCacher(file_path=str(self._cache_dir / "metrics.h5"))
        )


class MLFlowMetricDataCacher(MetricDataCacher):
    def __init__(self, experiment_name: str, tracking_uri: str, run_name: str):
        from atria_insights.storage.data_cachers._mlflow import MLflowDataCacher

        super().__init__(
            cacher=MLflowDataCacher(
                experiment_name=experiment_name,
                run_name=run_name,
                artifact_prefix="metrics",
                tracking_uri=tracking_uri,
            )
        )
