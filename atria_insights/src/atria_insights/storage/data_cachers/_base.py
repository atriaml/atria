import json
from abc import abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any, TypeVar

import h5py
import numpy as np
import torch
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa: F401
from atria_logger import get_logger

from atria_insights.storage.data_cachers._common import CacheData

logger = get_logger(__name__)

T = TypeVar("T")


class DataCacher:
    def __init__(self, file_path: str):
        self._file_path = Path(file_path)
        self._file_path.parent.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def sample_exists(self, sample_key: str) -> bool: ...

    @abstractmethod
    def save_sample(self, data: CacheData) -> None: ...

    @abstractmethod
    def load_sample(self, sample_key: str) -> CacheData: ...


class HDF5SampleCacher(DataCacher):
    def __init__(self, file_path: str):
        super().__init__(file_path=file_path)
        self._compression_kwargs = {"compression": "gzip", "compression_opts": 9}

    def _sample_group(self, hf: h5py.Group, sample_key: str) -> h5py.Group:
        return hf.create_group(sample_key) if sample_key not in hf else hf[sample_key]  # type: ignore

    def _create_or_update_dataset(
        self, hf: h5py.Group, data_key: str, data: np.ndarray
    ) -> None:
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        if data_key in hf:
            logger.warning(
                f"Dataset {data_key} already exists in HDF5 file. Overwriting."
            )
            hf[data_key][...] = data
        else:
            hf.create_dataset(data_key, data=data, **self._compression_kwargs)

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
            hf.attrs[key] = json.dumps(value)

    def _load_attrs(self, hf: h5py.Group) -> dict[str, Any]:
        attrs: dict[str, Any] = {}
        for key, value in hf.attrs.items():
            attrs[key] = json.loads(value)
        return attrs

    def _save_arrays(self, hf: h5py.Group, arrays: dict[str, Any]) -> None:
        for array_key, data in arrays.items():
            if data is None:
                continue

            logger.debug(f"Storing data_key: {array_key} of type {type(data)}")
            self._save_array(hf=hf, array_key=f"{array_key}", data=data)

    def _load_arrays(self, hf: h5py.Group) -> dict[str, Any]:
        arrays: dict[str, Any] = {}
        for array_key in hf.keys():
            dataset = hf[array_key]
            if isinstance(dataset, h5py.Dataset):
                arrays[array_key] = torch.from_numpy(dataset[()])
            elif isinstance(dataset, h5py.Group):
                arrays[array_key] = {}
                for sub_key in dataset.keys():
                    arrays[array_key][sub_key] = torch.from_numpy(dataset[sub_key][()])
            else:
                raise ValueError(f"Unexpected type {type(dataset)} for key {array_key}")

        return arrays

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
        with h5py.File(self._file_path, "r") as hf:
            return sample_key in hf

    def sample_sample(self, data: CacheData) -> None:
        with h5py.File(self._file_path, "a") as hf:
            # sample_key
            sample_key = data.sample_id
            assert sample_key is not None, "sample_id must be provided in data_dict."
            state_group = self._sample_group(hf, sample_key)

            # attrs
            if data.attrs is not None:
                logger.debug(f"Storing attrs for sample_key: {sample_key}")
                self._save_attrs(state_group, data.attrs)

            # data dict must provide attrs, arrays, and sample_key
            if data.tensors is not None:
                self._save_arrays(state_group, data.tensors)

            # log all keys in the h5 file
            logger.debug(f"Stored sample_key: {sample_key} in {self._file_path}")

    def load_sample(self, sample_key: str) -> CacheData:
        with h5py.File(self._file_path, "r") as hf:
            if sample_key not in hf:
                raise ValueError(f"Sample key {sample_key} not found in HDF5 file.")

            # get state group
            state_group = hf[sample_key]

            assert isinstance(state_group, h5py.Group), (
                f"Expected h5py.Group, got {type(state_group)}"
            )

            return CacheData(
                sample_id=sample_key,
                attrs=self._load_attrs(state_group),
                tensors=self._load_arrays(state_group),
            )
