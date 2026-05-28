"""Msgpack Storage Manager for handling dataset splits."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

import deltalake
import pandas as pd
from atria_logger import get_logger
from atria_types import BaseDataInstance, DatasetSplitType

from atria_datasets.core.dataset._datasets import SplitIterator
from atria_datasets.core.storage.shard_list_datasets import MsgpackShardListDataset

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


class DeltalakeStorageManager:
    """Manages deltalake storage for dataset splits with parallel writing support."""

    def __init__(
        self,
        storage_dir: str | Path,
        config_name: str,
        num_processes: int = 8,
        max_memory: int = 1000_000_000,
        max_shard_size: int = 100_000,
        name_suffix: str = "",
    ):
        self.storage_dir = Path(storage_dir)
        self.config_name = config_name
        self.num_processes = num_processes
        self.max_memory = max_memory
        self.max_shard_size = max_shard_size
        self.name_suffix = name_suffix

        self._setup_directories()

    def _setup_directories(self) -> None:
        """Create necessary directory structure."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        assert self.storage_dir.is_dir(), (
            f"Storage directory {self.storage_dir} must be a directory."
        )
        (self.storage_dir / self.config_name).mkdir(parents=True, exist_ok=True)

    def split_dir(self, split: DatasetSplitType) -> Path:
        """Get the directory path for a specific split."""
        split_dir = (
            self.storage_dir
            / self.config_name
            / "delta"
            / split.value
            / self.name_suffix
        )
        split_dir.mkdir(parents=True, exist_ok=True)
        return split_dir

    def dataset_exists(self) -> bool:
        """Check if the dataset exists in storage."""
        return (self.storage_dir / self.config_name / "delta").exists()

    def split_exists(self, split: DatasetSplitType) -> bool:
        return self.split_dir(split).exists()

    def get_splits(self) -> list[DatasetSplitType]:
        """Get all available splits in storage."""
        return [split for split in DatasetSplitType if self.split_exists(split)]

    def purge_split(self, split: DatasetSplitType) -> None:
        split_dir = self.split_dir(split)
        if split_dir.exists():
            logger.info(
                f"Purging dataset split {split.value} from storage {split_dir}."
            )
            shutil.rmtree(split_dir)

    def prepare_split_files(self, data_dir: str) -> set[tuple[str, str]]:
        # Get all delta files in the storage directory
        delta_files = list(
            (self.storage_dir / self.config_name / "delta").glob("**/*.*")
        )

        # Create a list of tuples (source, target) for delta files
        files_src_tgt = {
            (str(f), str(f.relative_to(self.storage_dir)))
            for f in delta_files
            if f.is_file()
        }

        def map_file_path(file_path):
            if pd.isna(file_path):
                return None  # skip NaN
            parsed = urlparse(file_path)
            path = parsed.path

            if path.startswith("shards/"):
                tgt = str(Path(self.config_name) / path)
                files_src_tgt.add(
                    (str(Path(self.storage_dir) / self.config_name / path), tgt)
                )
            else:
                tgt = str(Path(self.config_name) / path)
                files_src_tgt.add((str(Path(data_dir) / path), tgt))
            return tgt

        for split in list(DatasetSplitType):
            if not self.split_exists(split):
                continue

            dt = deltalake.DeltaTable(self.split_dir(split=split))
            all_columns = [f.name for f in dt.schema().fields]

            # Find all file_path columns (case-insensitive)
            file_path_cols = [col for col in all_columns if "file_path" in col.lower()]
            content_cols = [
                file_path_col.replace("file_path", "content")
                for file_path_col in file_path_cols
            ]
            dataframe = dt.to_pandas(columns=content_cols + file_path_cols)
            for file_path_column, content_column in zip(
                file_path_cols, content_cols, strict=True
            ):
                if not dataframe[content_column].dropna().empty:
                    continue
                file_path_col_data = dataframe[file_path_column].dropna()
                if file_path_col_data.empty:
                    continue
                file_path_col_data.apply(map_file_path)
        return files_src_tgt

    def write_split(self, split_iterator: SplitIterator) -> None:
        """Write a dataset split to storage with error handling."""
        try:
            self._write_split_internal(split_iterator)
        except (Exception, KeyboardInterrupt) as e:
            self.purge_split(split_iterator.split)
            error_msg = (
                "KeyboardInterrupt detected. Stopping dataset preparation..."
                if isinstance(e, KeyboardInterrupt)
                else f"Error while writing dataset split {split_iterator.split.value} to storage. Cleaning up..."
            )
            raise type(e)(error_msg) from e

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
