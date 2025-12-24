from collections import OrderedDict
from pathlib import Path
from typing import Any, TypeVar

import h5py
import numpy as np
import torch
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa: F401
from atria_logger import get_logger

from atria_insights.storage.data_cachers._base import DataCacher
from atria_insights.storage.data_cachers._common import SerializableSampleData

logger = get_logger(__name__)

T = TypeVar("T")


class HDF5DataCacher(DataCacher):
    def __init__(self, file_path: str | Path):
        super().__init__(file_path=file_path)

    def _sample_group(self, hf: h5py.Group, sample_key: str) -> h5py.Group:
        return hf.create_group(sample_key) if sample_key not in hf else hf[sample_key]  # type: ignore

    def _create_or_update_dataset(
        self, hf: h5py.Group, data_key: str, data: np.ndarray
    ) -> None:
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        if data_key not in hf:
            if isinstance(data, np.ndarray) and len(data.shape) > 1:
                hf.create_dataset(
                    data_key, data=data, compression="gzip", compression_opts=9
                )
            else:
                hf.create_dataset(data_key, data=data)
        else:
            logger.warning(
                f"Dataset {data_key} already exists in HDF5 file for {hf.name}. Overwriting..."
            )
            try:
                hf[data_key][...] = data
            except Exception as e:
                logger.error(
                    f"Failed to overwrite dataset {data_key} with data: {data} in HDF5 file for {hf.name}."
                )
                raise e

    def _save_array(
        self,
        hf: h5py.Group,
        array_key: str,
        data: np.ndarray | dict[str, np.ndarray] | OrderedDict[str, np.ndarray],
    ) -> None:
        if isinstance(data, np.ndarray | torch.Tensor):
            self._create_or_update_dataset(hf, array_key, data=data)
        elif isinstance(data, OrderedDict | dict):
            for key, value in data.items():
                self._create_or_update_dataset(hf, array_key + "/" + key, data=value)
        else:
            raise ValueError(f"data must be np.ndarray or dict, got {type(data)}")

    def _save_attrs(self, hf: h5py.Group, attrs: dict[str, Any]) -> None:
        for key, value in attrs.items():
            if value is None:
                continue
            hf.attrs[key] = value

    def _load_attrs(self, hf: h5py.Group) -> dict[str, Any]:
        attrs: dict[str, Any] = {}
        for key, value in hf.attrs.items():
            attrs[key] = value
        return attrs

    def _save_tensors(self, hf: h5py.Group, tensors: dict[str, Any]) -> None:
        for array_key, data in tensors.items():
            if data is None:
                continue

            logger.debug(f"Storing data_key: {array_key} of type {type(data)}")
            self._save_array(hf=hf, array_key=f"{array_key}", data=data)

    def _load_tensor_value(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value)
        else:
            return torch.tensor(value)

    def _load_tensors(self, hf: h5py.Group) -> dict[str, Any]:
        tensors: dict[str, Any] = {}
        for array_key in hf.keys():
            dataset = hf[array_key]
            if isinstance(dataset, h5py.Dataset):
                tensors[array_key] = self._load_tensor_value(dataset[()])
            elif isinstance(dataset, h5py.Group):
                tensors[array_key] = {}
                for sub_key in dataset.keys():
                    tensors[array_key][sub_key] = self._load_tensor_value(
                        dataset[sub_key][()]
                    )
            else:
                raise ValueError(f"Unexpected type {type(dataset)} for key {array_key}")

        return tensors

    def format_tree(self, h5obj: h5py.Group, indent=0):
        lines = []
        prefix = "  " * indent

        for attr_key, attr_val in h5obj.attrs.items():
            lines.append(f"{prefix}@{attr_key} = {attr_val}")

        for key, item in h5obj.items():
            if isinstance(item, h5py.Group):
                lines.append(f"{prefix}{key}/")
                lines.append(self.format_tree(item, indent + 1))

            elif isinstance(item, h5py.Dataset):
                shape = item.shape if item.shape != () else "scalar"
                dtype = item.dtype
                lines.append(f"{prefix}{key}  shape={shape}  dtype={dtype}")

        return "\n".join(lines)

    def sample_exists(self, sample_key: str) -> bool:
        if not Path(self._file_path).exists():
            return False

        with h5py.File(self._file_path, "r") as hf:
            return sample_key in hf

    def list_sample_keys(self) -> list[str]:
        logger.debug(f"Listing sample keys in HDF5 file: {self._file_path}")
        with h5py.File(self._file_path, "r") as hf:
            return list(hf.keys())

    def save_sample(self, data: SerializableSampleData) -> None:
        with h5py.File(self._file_path, "a") as hf:
            # sample_key
            sample_key = data.sample_id
            assert sample_key is not None, "sample_id must be provided in data_dict."
            state_group = self._sample_group(hf, sample_key)

            # attrs
            if data.attrs is not None:
                logger.debug(
                    f"Storing attrs for sample_key: {sample_key}: {data.attrs}"
                )
                self._save_attrs(state_group, data.attrs)

            # data dict must provide attrs, tensors, and sample_key
            if data.tensors is not None:
                self._save_tensors(state_group, data.tensors)

            # log all keys in the h5 file
            logger.debug(f"Stored sample_key: {sample_key} in {self._file_path}")

    def load_sample(self, sample_key: str) -> SerializableSampleData:
        with h5py.File(self._file_path, "r") as hf:
            if sample_key not in hf:
                raise ValueError(f"Sample key {sample_key} not found in HDF5 file.")

            # get state group
            state_group = hf[sample_key]

            assert isinstance(state_group, h5py.Group), (
                f"Expected h5py.Group, got {type(state_group)}"
            )

            return SerializableSampleData(
                sample_id=sample_key,
                attrs=self._load_attrs(state_group),
                tensors=self._load_tensors(state_group),
            )
