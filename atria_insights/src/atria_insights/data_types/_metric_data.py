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
    data: dict[str, float | torch.Tensor | str]


class BatchMetricData(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )

    sample_id: list[str]
    data: dict[str, torch.Tensor | list[float | str]]

    def tolist(self) -> list[SampleMetricData]:
        metric_data_list = []
        batch_size = len(self.sample_id)

        metric_data = self.data
        for key, value in metric_data.items():
            assert isinstance(value, list | torch.Tensor), (
                f"Data for key '{key}' must be a list or torch.Tensor, got {type(value)}."
            )
            assert len(value) == batch_size, (
                f"Data for key '{key}' has length {len(value)}, expected {batch_size}."
            )

        # convert dict of lists to list of dicts
        list_of_dicts = [
            {key: value[i] for key, value in metric_data.items()}
            for i in range(batch_size)
        ]

        # convert dict of lists to list of
        metric_data_list = [
            SampleMetricData(
                sample_id=self.sample_id[sample_idx], data=list_of_dicts[sample_idx]
            )
            for sample_idx in range(batch_size)
        ]
        return metric_data_list

    @classmethod
    def fromlist(cls, metric_data_list: list[SampleMetricData]) -> BatchMetricData:
        if not metric_data_list:
            raise ValueError("metric_data_list must not be empty")

        sample_ids = [
            sample_metric_data.sample_id for sample_metric_data in metric_data_list
        ]

        # make list of dicts
        list_of_dicts = [
            sample_metric_data.data for sample_metric_data in metric_data_list
        ]

        # convert list of dicts to dict of lists
        data_dict: dict[str, list[Any]] = {}
        for key in list_of_dicts[0].keys():
            data_dict[key] = [d[key] for d in list_of_dicts]

        # validate all values for each key are of the same type
        for key, value_list in data_dict.items():
            first_type = type(value_list[0])
            first_shape = (
                value_list[0].shape if isinstance(value_list[0], torch.Tensor) else None
            )
            if not all(isinstance(v, first_type) for v in value_list):
                raise ValueError(
                    f"All values for key '{key}' must be of the same type. "
                    f"Found types: {[type(v) for v in value_list]}"
                )
            if isinstance(value_list[0], torch.Tensor):
                if not all((v.shape == first_shape) for v in value_list):
                    raise ValueError(
                        f"All tensors/arrays for key '{key}' must have the same shape. "
                        f"Expected shape {first_shape}, found shapes: {[v.shape for v in value_list]}"
                    )

        # stack tensors where applicable
        for key, value_list in data_dict.items():
            if isinstance(value_list[0], torch.Tensor):
                if value_list[0].ndim == 0:
                    data_dict[key] = torch.tensor(value_list)
                elif value_list[0].ndim == 1 and value_list[0].shape[0] == 1:
                    # special case for 1D tensors of shape (1,)
                    data_dict[key] = torch.cat(value_list)
                else:
                    data_dict[key] = torch.stack(value_list)

            assert len(data_dict[key]) == len(sample_ids), (
                f"Data for key '{key}' has length {len(data_dict[key])}, expected {len(sample_ids)}."
            )
        return cls(sample_id=sample_ids, data=data_dict)

    @property
    def batch_size(self) -> int:
        """Return the batch size of the metric data."""
        return len(self.sample_id)
