from __future__ import annotations

from typing import Any

import torch
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa: F401
from atria_logger import get_logger
from pydantic import BaseModel, ConfigDict

BaselineType = torch.Tensor | tuple[torch.Tensor]

logger = get_logger(__name__)


class SampleMetricData(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )

    sample_id: str
    data: dict[str, Any]


class BatchMetricData(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )

    sample_id: list[str]
    data: dict[str, Any]

    def tolist(self) -> list[SampleMetricData]:
        metric_data_list = []
        batch_size = len(self.sample_id)
        for i in range(batch_size):
            sample_data = {}
            for key, value in self.data.items():
                if not (isinstance(value, list) or isinstance(value, torch.Tensor)):
                    raise ValueError(
                        "All values in data must be either list or torch.Tensor for batching."
                    )
                sample_data[key] = value[i]
            sample_data = {
                key: value[i]
                for key, value in self.data.items()
                if isinstance(value, list) or isinstance(value, torch.Tensor)
            }
            metric_data_list.append(
                SampleMetricData(sample_id=self.sample_id[i], data=sample_data)
            )
        return metric_data_list

    @property
    def batch_size(self) -> int:
        """Return the batch size of the metric data."""
        return len(self.sample_id)
