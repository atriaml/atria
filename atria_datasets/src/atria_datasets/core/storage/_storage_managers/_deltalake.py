"""Deltalake Storage Manager for handling dataset splits."""

from __future__ import annotations

import itertools
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Self
from urllib.parse import urlparse

import deltalake
import pandas as pd
import pyarrow as pa
import ray
import tqdm
from atria_logger import get_logger
from atria_types import BaseDataInstance, DatasetSplitType

from atria_datasets.core.dataset._datasets import SplitIterator
from atria_datasets.core.dataset._split_iterators import InstanceTransform

logger = get_logger(__name__)

_RAY_RUNTIME_ENV = {"env_vars": {"PYTHONPATH": ":".join(sys.path)}}


class DeltalakeWriterWorker:
    """Writes WebDataset tar shards for binary content and accumulates row dicts for delta lake."""

    def __init__(
        self,
        worker_id: int,
        write_dir: Path,
        split: str,
        data_model: type[BaseDataInstance],
        max_shard_size: int,
        preprocess_transform: InstanceTransform | None = None,
    ):
        self._worker_id = worker_id
        self._write_dir = Path(write_dir)
        self._split = split
        self._data_model = data_model
        self._max_shard_size = max_shard_size
        self._preprocess_transform = preprocess_transform
        self._wds_writer: Any = None
        self._current_shard = 0
        self._current_shard_path: Path | None = None
        self._offset = 0
        self._accumulated_rows: list[dict] = []

    def load(self) -> Self:
        return self

    def _ensure_wds_writer(self) -> None:
        import webdataset as wds

        if self._wds_writer is None:
            file_path = (
                self._write_dir
                / "shards"
                / self._split
                / f"{self._worker_id:06d}-%06d.tar"
            )
            file_path.parent.mkdir(parents=True, exist_ok=True)
            self._wds_writer = wds.ShardWriter(
                str(file_path), maxcount=self._max_shard_size, verbose=0
            )
            self._current_shard = self._wds_writer.shard
            self._current_shard_path = (
                Path("shards") / self._split / Path(self._wds_writer.fname).name
            )

    def write(self, index: int, sample: Any) -> None:
        if self._preprocess_transform is not None:
            result = self._preprocess_transform(index, sample)
            list_of_samples = result if isinstance(result, list) else [result]
        else:
            list_of_samples = [sample]

        for s in list_of_samples:
            # Detect shard rotation and update tracked path/offset
            if (
                self._wds_writer is not None
                and self._current_shard != self._wds_writer.shard
            ):
                self._current_shard = self._wds_writer.shard
                self._current_shard_path = (
                    Path("shards") / self._split / Path(self._wds_writer.fname).name
                )
                self._offset = 0

            sample_row = s.to_row()

            for key in list(sample_row.keys()):
                if "file_path" not in key:
                    continue
                content_key = key.replace("file_path", "content")
                assert content_key in sample_row, (
                    f"Column '{content_key}' not found in sample row. Expected 'content' "
                    f"column for the corresponding file path '{key}'."
                )
                content = sample_row[content_key]
                if content is None and sample_row[key] is None:
                    continue
                assert content is not None, (
                    f"Content for key '{content_key}' is None. Expected valid binary content."
                )
                self._ensure_wds_writer()
                assert self._wds_writer is not None
                self._wds_writer.write({"__key__": str(s.key), content_key: content})
                member = self._wds_writer.tarstream.tarstream.members[-1]
                sample_row[key] = (
                    f"{self._current_shard_path}?offset={self._offset + 1536}&length={member.size}"
                )
                sample_row[content_key] = None
                self._offset = self._wds_writer.tarstream.tarstream.offset

            self._accumulated_rows.append(sample_row)

    def close(self) -> list[dict]:
        if self._wds_writer is not None:
            self._wds_writer.close()
            self._wds_writer = None
        return self._accumulated_rows


@ray.remote
class DeltalakeShardWriterActor:
    def __init__(
        self,
        worker_id: int,
        write_dir: Path,
        split: str,
        data_model: type[BaseDataInstance],
        max_shard_size: int,
        preprocess_transform: InstanceTransform | None = None,
    ):
        self.writer = DeltalakeWriterWorker(
            worker_id=worker_id,
            write_dir=write_dir,
            split=split,
            data_model=data_model,
            max_shard_size=max_shard_size,
            preprocess_transform=preprocess_transform,
        ).load()

    def write(self, sample_tuple: tuple) -> bool:
        idx, sample = sample_tuple
        try:
            self.writer.write(idx, sample)
        except Exception:
            logger.exception(f"Error writing sample at index {idx}")
        return True

    def close(self) -> list[dict]:
        return self.writer.close()


def _write_rows_to_deltalake(
    rows: list[dict],
    split_dir: Path,
    data_model: type[BaseDataInstance],
    mode: str = "overwrite",
) -> None:
    table = pa.Table.from_pylist(rows, schema=data_model.pa_schema())
    deltalake.write_deltalake(str(split_dir), table, mode=mode)  # type: ignore[call-overload]


class RayParallelDeltalakeWriter:
    def __init__(
        self,
        write_dir: Path,
        num_workers: int = 4,
        max_shard_size: int = 100_000,
        max_concurrent_tasks_limit: int = 128,
        max_memory_per_actor: int = 500 * 1024 * 1024,
    ):
        self.write_dir = write_dir
        self.num_workers = num_workers
        self.max_shard_size = max_shard_size
        self._max_concurrent_tasks_limit = max_concurrent_tasks_limit
        self._max_memory_per_actor = max_memory_per_actor

    def write_split(self, split_iterator: SplitIterator, split_dir: Path) -> None:
        split_name = split_iterator.split.value
        logger.info(f"Writing split {split_name} with {self.num_workers} Ray actors...")

        ray.init(
            num_cpus=self.num_workers,
            local_mode=self.num_workers == 1,
            runtime_env=_RAY_RUNTIME_ENV,
        )

        actors = [
            DeltalakeShardWriterActor.options(memory=self._max_memory_per_actor).remote(
                worker_id=i,
                write_dir=self.write_dir,
                split=split_name,
                data_model=split_iterator.data_model,
                max_shard_size=self.max_shard_size,
                preprocess_transform=split_iterator._tf,
            )
            for i in range(self.num_workers)
        ]

        try:
            data_iterator = iter(split_iterator)
            pending_tasks = []
            actor_iterator = itertools.cycle(actors)

            for idx, sample in tqdm.tqdm(
                data_iterator, desc=f"Writing split {split_name}"
            ):
                actor = next(actor_iterator)
                pending_tasks.append(actor.write.remote((idx, sample)))  # type: ignore[union-attr]

                if len(pending_tasks) >= self._max_concurrent_tasks_limit:
                    ready_tasks, pending_tasks = ray.wait(pending_tasks, num_returns=1)
                    try:
                        ray.get(ready_tasks)
                    except Exception as e:
                        logger.exception("Error in shard writer actor task")
                        raise e

            ray.get(pending_tasks)

            all_actor_rows = ray.get([actor.close.remote() for actor in actors])  # type: ignore[union-attr]
            all_rows = [row for actor_rows in all_actor_rows for row in actor_rows]

            if all_rows:
                logger.info(
                    f"Writing {len(all_rows)} rows to delta lake at {split_dir}"
                )
                _write_rows_to_deltalake(all_rows, split_dir, split_iterator.data_model)

        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt detected, shutting down Ray actors...")
            raise
        except Exception as e:
            logger.exception("Error during Ray parallel split writing")
            raise e
        finally:
            if ray.is_initialized():
                ray.shutdown()


class SingleDeltalakeWriter:
    def __init__(self, write_dir: Path, max_shard_size: int = 100_000):
        self.write_dir = write_dir
        self.max_shard_size = max_shard_size

    def write_split(self, split_iterator: SplitIterator, split_dir: Path) -> None:
        split_name = split_iterator.split.value
        data_iterator = iter(split_iterator)

        worker = DeltalakeWriterWorker(
            worker_id=0,
            write_dir=self.write_dir,
            split=split_name,
            data_model=split_iterator.data_model,
            max_shard_size=self.max_shard_size,
            preprocess_transform=split_iterator._tf,
        ).load()

        for idx, sample in tqdm.tqdm(data_iterator, desc=f"Writing split {split_name}"):
            try:
                worker.write(idx, sample)
            except Exception:
                logger.exception(f"Error writing sample at index {idx}")

        all_rows = worker.close()

        if all_rows:
            logger.info(f"Writing {len(all_rows)} rows to delta lake at {split_dir}")
            _write_rows_to_deltalake(all_rows, split_dir, split_iterator.data_model)


class DeltalakeStorageManager:
    """Manages delta lake storage for dataset splits."""

    def __init__(
        self,
        storage_dir: str | Path,
        config_name: str,
        num_processes: int = 8,
        max_memory: int = 1_000_000_000,
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
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        assert self.storage_dir.is_dir(), (
            f"Storage directory {self.storage_dir} must be a directory."
        )
        (self.storage_dir / self.config_name).mkdir(parents=True, exist_ok=True)

    def split_dir(self, split: DatasetSplitType) -> Path:
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
        return (self.storage_dir / self.config_name / "delta").exists()

    def split_exists(self, split: DatasetSplitType) -> bool:
        return (self.split_dir(split) / "_delta_log").exists()

    def get_splits(self) -> list[DatasetSplitType]:
        return [split for split in DatasetSplitType if self.split_exists(split)]

    def purge_split(self, split: DatasetSplitType) -> None:
        split_dir = self.split_dir(split)
        if split_dir.exists():
            logger.info(
                f"Purging dataset split {split.value} from storage {split_dir}."
            )
            shutil.rmtree(split_dir)

    def write_split(self, split_iterator: SplitIterator) -> None:
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

    def _write_split_internal(self, split_iterator: SplitIterator) -> None:
        split_dir = self.split_dir(split_iterator.split)
        write_dir = self.storage_dir / self.config_name
        logger.info(
            f"Writing dataset split {split_iterator.split.value} to {split_dir} "
            f"({'parallel' if self.num_processes > 1 else 'single'} mode)"
        )

        if self.num_processes > 1:
            writer = RayParallelDeltalakeWriter(
                write_dir=write_dir,
                num_workers=self.num_processes,
                max_shard_size=self.max_shard_size,
            )
        else:
            writer = SingleDeltalakeWriter(
                write_dir=write_dir, max_shard_size=self.max_shard_size
            )

        split_iterator.disable_tf()
        try:
            writer.write_split(split_iterator, split_dir)
        finally:
            split_iterator.enable_tf()

    def read_split(
        self,
        split: DatasetSplitType,
        data_model: type[BaseDataInstance],
        output_transform: Callable | None = None,
        allowed_keys: set[str] | None = None,
        streaming_mode: bool = False,
    ) -> SplitIterator:
        from atria_datasets.core.storage.deltalake_reader import (
            InMemoryDeltalakeReader,
            LocalDeltalakeReader,
        )

        if not self.split_exists(split):
            raise RuntimeError(
                f"Dataset split {split.value} not prepared. Please call `write_split()` first."
            )

        if allowed_keys is not None:
            allowed_keys = allowed_keys.copy()
            allowed_keys.update({"sample_id", "index"})

        reader_cls = LocalDeltalakeReader if streaming_mode else InMemoryDeltalakeReader
        base_iterator = reader_cls(
            table_path=str(self.split_dir(split=split)),
            data_model=data_model,
            allowed_keys=allowed_keys,
            storage_dir=str(self.storage_dir),
            config_name=self.config_name,
        )

        return SplitIterator(
            split=split,
            base_iterator=base_iterator,
            output_transform=output_transform,
            data_model=data_model,
        )

    def prepare_split_files(self, data_dir: str) -> set[tuple[str, str]]:
        delta_files = list(
            (self.storage_dir / self.config_name / "delta").glob("**/*.*")
        )
        files_src_tgt = {
            (str(f), str(f.relative_to(self.storage_dir)))
            for f in delta_files
            if f.is_file()
        }

        def map_file_path(file_path):
            if pd.isna(file_path):
                return None
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

            file_path_cols = [col for col in all_columns if "file_path" in col.lower()]
            content_cols = [
                col.replace("file_path", "content") for col in file_path_cols
            ]
            dataframe = dt.to_pandas(columns=content_cols + file_path_cols)
            for file_path_col, content_col in zip(
                file_path_cols, content_cols, strict=True
            ):
                if not dataframe[content_col].dropna().empty:
                    continue
                file_path_col_data = dataframe[file_path_col].dropna()
                if file_path_col_data.empty:
                    continue
                file_path_col_data.apply(map_file_path)

        return files_src_tgt
