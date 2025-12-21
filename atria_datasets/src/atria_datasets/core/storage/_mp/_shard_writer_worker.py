"""
Shard Writer Module

Provides a single-process OnlineShardWriter class and a Ray actor wrapper for distributed
writing of dataset samples to storage shards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import ray
import webdataset as wds
from atria_logger import get_logger
from atria_types import DatasetShardInfo

from atria_datasets.core.dataset._split_iterators import InstanceTransform
from atria_datasets.core.storage._shard_writers._msgpack import MsgpackShardWriter
from atria_datasets.core.storage.utilities import FileStorageType

if TYPE_CHECKING:
    from atria_types import BaseDataInstance

    from atria_datasets.core.storage.utilities import FileStorageType


logger = get_logger(__name__)


class ShardWriterWorker:
    def __init__(
        self,
        data_model: type[BaseDataInstance],
        storage_type: FileStorageType,
        storage_file_pattern: str,
        max_shard_size: int,
        preprocess_transform: InstanceTransform | None = None,
    ):
        self._data_model = data_model
        self._storage_type = storage_type
        self._storage_file_pattern = storage_file_pattern
        self._max_shard_size = max_shard_size
        self._preprocess_transform = preprocess_transform
        self._writer: MsgpackShardWriter | wds.writer.ShardWriter | None = None
        self._write_info: list[DatasetShardInfo] = []

    def load(self) -> Self:
        """Initializes the underlying shard writer based on the storage type."""
        if self._storage_type == FileStorageType.MSGPACK:
            self._writer = MsgpackShardWriter(
                self._storage_file_pattern,
                maxcount=self._max_shard_size,
                overwrite=True,
            )
        elif self._storage_type == FileStorageType.WEBDATASET:
            self._writer = wds.writer.ShardWriter(
                self._storage_file_pattern, maxcount=self._max_shard_size
            )
        else:
            raise ValueError(
                f"Unsupported storage type: {self._storage_type}. Supported types are: "
                f"{FileStorageType.MSGPACK}, {FileStorageType.WEBDATASET}"
            )
        assert self._writer.fname is not None, "Writer filename is None."
        self._write_info.append(DatasetShardInfo(url=self._writer.fname))
        return self

    def write(self, index: int, sample: BaseDataInstance):
        """Writes a single dataset sample to the shard."""
        import webdataset as wds

        if self._writer is None:
            raise RuntimeError("ShardWriter is not loaded. Call `load()` first.")
        assert self._writer.fname is not None, "Writer filename is None."

        # Preprocess sample if needed
        if self._preprocess_transform:
            list_of_samples = self._preprocess_transform(index, sample)
            if not isinstance(list_of_samples, list):
                list_of_samples = [list_of_samples]
        else:
            list_of_samples = [sample]

        # Detect new shard creation and update info
        if self._writer.shard != self._write_info[-1].shard:
            self._write_info.append(
                DatasetShardInfo(
                    url=self._writer.fname,
                    shard=self._writer.shard,
                    nsamples=self._writer.count,
                    filesize=self._writer.size,
                )
            )

        # Write sample
        for sample in list_of_samples:
            if isinstance(self._writer, wds.writer.ShardWriter):
                output_sample = {"__key__": str(sample.key)}
                for key, value in sample.model_dump().items():
                    if value is None:
                        continue
                    output_sample[
                        f"{key}.txt" if isinstance(value, str) else f"{key}.mp"
                    ] = value
                self._writer.write(output_sample)
            else:
                self._writer.write({**sample.model_dump(), "key": sample.key})

        # Update current shard info
        shard_info = self._write_info[-1]
        shard_info.url = self._writer.fname
        shard_info.shard = self._writer.shard
        shard_info.nsamples = self._writer.count
        shard_info.filesize = self._writer.size

    def close(self) -> list[DatasetShardInfo]:
        """Finalizes the writer and returns shard metadata."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        return self._write_info


@ray.remote
class ShardWriterWorkerActor:
    """Ray actor wrapper around the single-process ShardWriter."""

    def __init__(self, *args, **kwargs):
        self._writer = ShardWriterWorker(*args, **kwargs)

    def load(self):
        return self._writer.load()

    def write(self, *args, **kwargs):
        return self._writer.write(*args, **kwargs)

    def close(self):
        return self._writer.close()
