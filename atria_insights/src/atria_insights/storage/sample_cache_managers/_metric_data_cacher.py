import json
from pathlib import Path

import torch
from atria_logger import get_logger

from atria_insights.data_types._metric_data import SampleMetricData
from atria_insights.explainability_metrics._base import ExplainabilityMetricConfig
from atria_insights.storage.data_cachers._common import SerializableSampleData
from atria_insights.storage.sample_cache_managers._base import BaseSampleCacheManager
from atria_insights.storage.sample_cache_managers._utilities import to_serializable

logger = get_logger(__name__)


class MetricDataCacher(BaseSampleCacheManager[SampleMetricData]):
    def __init__(
        self,
        cache_dir: str | Path,
        config: ExplainabilityMetricConfig,
        file_name: str | None = None,
    ):
        # create a child cache dir for the given explainer

        file_name = f"metrics/{file_name}.hdf5"
        if not (Path(cache_dir) / file_name).exists():
            # find all files starting with the name
            existing_files = list(
                (Path(cache_dir) / "metrics").glob(f"{config.type}-*.hdf5")
            )
            # go over files to see to take the largest file
            if existing_files:
                existing_files = sorted(
                    existing_files, key=lambda x: x.stat().st_size, reverse=True
                )
                file_name = existing_files[0].relative_to(cache_dir)

            # file_name = f"metrics/{config.type}-{config.hash}.hdf5"
        super().__init__(cache_dir=Path(cache_dir), file_name=file_name)
        self._config = config

        # save config to attrs
        # self._dump_config()

    def _dump_config(self) -> dict:
        if not self.file_path.with_suffix(".yaml").exists():
            self.save_file_attrs({"config": json.dumps(self._config.to_dict())})
            with open(self.file_path.with_suffix(".yaml"), "w") as f:
                f.write(self._config.to_yaml())

    def _serialize_type(self, data: SampleMetricData) -> SerializableSampleData:
        # find all tensors in data
        tensors = {}
        attrs = {}
        for key, value in data.data.items():
            if isinstance(value, torch.Tensor):
                tensors[key] = value
            else:
                attrs[key] = to_serializable(value)

        return SerializableSampleData(
            sample_id=data.sample_id,
            attrs={
                "sample_id": data.sample_id,
                "config_hash": self._config.hash,
                **attrs,
            },
            tensors=tensors,
        )

    def _deserialize_type(self, data: SerializableSampleData) -> SampleMetricData:
        assert data.attrs is not None, "attrs must be provided in CacheData."
        assert data.tensors is not None, "tensors must be provided in CacheData."

        attrs = data.attrs
        sample_id = attrs.pop("sample_id", None)
        assert isinstance(sample_id, str), "sample_id must be a string."
        attrs.pop("config_hash", None)

        return SampleMetricData(
            sample_id=sample_id,
            data={**attrs, **data.tensors},  # type: ignore
        )
