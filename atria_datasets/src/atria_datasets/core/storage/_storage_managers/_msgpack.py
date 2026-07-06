"""Msgpack Storage Manager for handling dataset splits."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

from atria_logger import get_logger
from atria_types import BaseDataInstance, DatasetSplitType

from atria_datasets.core.dataset._datasets import SplitIterator
from atria_datasets.core.storage._shard_list_datasets import MsgpackShardListDataset
from atria_datasets.core.storage._storage_managers._base import StorageManager

logger = get_logger(__name__)


class MsgpackStorageReadTransform:
    """Transform for reading and filtering msgpack storage samples."""

    def __init__(
        self, data_model: type[BaseDataInstance], allowed_keys: set[str] | None = None
    ):
        self.data_model = data_model
        self.allowed_keys = allowed_keys

    def __call__(self, sample: dict) -> BaseDataInstance:
        filtered_sample = {}
        for key in list(sample.keys()):
            if self.allowed_keys is not None and key not in self.allowed_keys:
                continue
            filtered_sample[key] = sample[key]
        return self.data_model(**filtered_sample)


class MsgpackStorageManager(StorageManager):
    """Manages msgpack storage for dataset splits with parallel writing support."""

    storage_prefix: ClassVar[str] = "msgpack"

    def __init__(
        self,
        data_dir: str | Path,
        storage_dir: str | Path,
        config_name: str,
        num_processes: int = 8,
        max_shard_size: int = 100_000,
        name_suffix: str = "",
    ):
        self.max_shard_size = max_shard_size
        super().__init__(
            data_dir=data_dir,
            storage_dir=storage_dir,
            config_name=config_name,
            num_processes=num_processes,
            name_suffix=name_suffix,
        )

    def split_exists(self, split: DatasetSplitType) -> bool:
        """Check if a specific split exists in storage."""
        offsets = list(self.split_dir(split).glob("*.msgpack.offsets"))
        return len(self.split_files(split)) > 0 and len(offsets) > 0

    def split_files(self, split: DatasetSplitType) -> list[Path]:
        """Get all msgpack files for a specific split."""
        return list(self.split_dir(split).glob("*.msgpack"))

    def _write_split_internal(self, split_iterator: SplitIterator) -> Path:
        """Internal method to write split data."""
        from atria_datasets.core.storage._mp._split_writers import (
            RayParallelSplitWriter,
            SingleSplitWriter,
        )

        split_dir = self.split_dir(split_iterator.split)
        logger.info(
            f"Writing dataset split {split_iterator.split.value} to {split_dir} "
            f"({'parallel' if self.num_processes > 1 else 'single'} mode)"
        )

        # Choose writer based on process count
        if self.num_processes > 1:
            writer = RayParallelSplitWriter(
                num_workers=self.num_processes, max_shard_size=self.max_shard_size
            )
        else:
            writer = SingleSplitWriter(max_shard_size=self.max_shard_size)

        # Disable transforms during writing
        split_iterator.disable_tf()
        try:
            write_info = writer.write_split(split_iterator, split_dir)
            self._log_write_results(write_info, split_iterator.split)
        finally:
            split_iterator.enable_tf()

        return split_dir

    def _log_write_results(self, write_info: list, split: DatasetSplitType) -> None:
        """Log the results of writing operation."""
        total_samples = sum(shard.nsamples for shard in write_info)
        logger.info(
            f"Successfully wrote {total_samples} samples to {len(write_info)} shards "
            f"for split {split.value}"
        )

    def read_split(
        self,
        split: DatasetSplitType,
        data_model: type[BaseDataInstance],
        output_transform: Callable | None = None,
        allowed_keys: set[str] | None = None,
    ) -> SplitIterator:
        """Read a dataset split from storage."""
        if not self.split_exists(split):
            raise RuntimeError(
                f"Dataset split {split.value} not prepared. Please call `write_split()` first."
            )

        # Ensure required keys are included
        if allowed_keys is not None:
            allowed_keys = allowed_keys.copy()  # Don't modify original set
            allowed_keys.update({"sample_id", "index"})

        return SplitIterator(
            split=split,
            base_iterator=MsgpackShardListDataset(self.split_files(split)),
            input_transform=MsgpackStorageReadTransform(
                data_model=data_model, allowed_keys=allowed_keys
            ),
            output_transform=output_transform,
            data_model=data_model,
        )
