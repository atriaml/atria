from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import tqdm
from atria_logger import get_logger
from atria_types._datasets import DatasetShardInfo

from atria_datasets.core.storage._mp._shard_writer_worker import (
    ShardWriterWorker,
    finalize_worker,
    init_writer,
    write_one,
)
from atria_datasets.core.storage._shard_writers._msgpack import DuplicateKeyError
from atria_datasets.core.storage.utilities import FileStorageType

if TYPE_CHECKING:
    from atria_types import BaseDataInstance, DatasetSplitType

    from atria_datasets.core.dataset._datasets import SplitIterator


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


def write_split_streaming(
    split_iterator, split_dir: Path, num_workers: int = 4, max_shard_size: int = 100_000
):
    split_dir.mkdir(parents=True, exist_ok=True)
    shard_infos = []
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(
        processes=num_workers,
        initializer=init_writer,
        initargs=(
            split_dir,
            split_iterator.data_model,
            max_shard_size,
            split_iterator._tf,
        ),
    )

    try:
        # streaming, no batching
        for _ in tqdm.tqdm(
            pool.imap_unordered(write_one, split_iterator, chunksize=1),
            total=len(split_iterator),
            desc="Processing samples",
        ):
            pass

        # Finalize workers
        shard_infos_per_worker = pool.map(finalize_worker, range(num_workers))
        shard_infos = [
            info for worker_infos in shard_infos_per_worker for info in worker_infos
        ]

    except Exception as e:
        logger.error(f"Error during shard writing: {e}")
        pool.terminate()
        pool.join()
        raise

    finally:
        pool.close()
        pool.join()

    # flatten list of lists
    shard_infos = [
        info for worker_infos in shard_infos_per_worker for info in worker_infos
    ]

    return shard_infos


def write_split_parallel(
    split_iterator: SplitIterator,
    split_dir: Path,
    max_shard_size: int = 100_000,
    num_processes: int = 8,
) -> list[DatasetShardInfo]:
    split_dir.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(
        processes=num_processes,
        initializer=init_writer,
        initargs=(
            split_dir,
            split_iterator.data_model,
            max_shard_size,
            split_iterator._tf,
        ),
    )

    try:
        # streaming, no batching
        for _ in tqdm.tqdm(
            pool.imap_unordered(write_one, split_iterator, chunksize=1),
            desc="Processing samples",
        ):
            pass

        # Finalize workers
        write_infos_per_worker: list[list[DatasetShardInfo]] = pool.map(
            finalize_worker, range(num_processes)
        )

        # Flatten the list of shard infos
        write_infos: list[DatasetShardInfo] = [
            info for worker_infos in write_infos_per_worker for info in worker_infos
        ]

    except Exception as e:
        logger.error(f"Error during shard writing: {e}")

        pool.terminate()
        pool.join()

        raise

    finally:
        if pool is not None:
            try:
                pool.close()
                pool.join()
            except Exception:
                pool.terminate()
                pool.join()
    return write_infos


def write_split_single(
    split_iterator: SplitIterator, split_dir: Path, max_shard_size: int = 100_000
) -> list[DatasetShardInfo]:
    split_name: str = split_iterator.split.value
    data_iterator = iter(split_iterator)

    writer = ShardWriterWorker(
        data_model=split_iterator.data_model,
        storage_type=FileStorageType.MSGPACK,
        storage_file_pattern=str(split_dir / "000000-%06d.msgpack"),
        max_shard_size=max_shard_size,
        preprocess_transform=split_iterator._tf,
    )
    writer.load()

    for idx, sample in tqdm.tqdm(
        data_iterator,
        desc=f"Writing split {split_name}",
        total=len(split_iterator) if hasattr(split_iterator, "__len__") else None,
    ):
        try:
            writer.write(idx, sample)
        except DuplicateKeyError:
            logger.warning(
                f"Duplicate key detected at sample index {idx}. Skipping sample."
            )

    write_info: list[DatasetShardInfo] = writer.close()

    # Filter out empty shards and renumber
    write_info = [shard for shard in write_info if shard.nsamples > 0]
    for i, shard in enumerate(write_info):
        shard.shard = i + 1

    return write_info


class MsgpackStorageManager:
    """
    Manager for storing datasets in MessagePack format with support for both
    single-process and multi-process writing.

    Args:
        storage_dir: Directory where the dataset shards will be stored
        config_name: Name/identifier for the dataset configuration
        num_processes: Number of processes to use for parallel writing (1 = single-process)
        max_memory: Maximum memory to use (currently unused but kept for compatibility)
        max_shard_size: Maximum number of samples per shard file
    """

    def __init__(
        self,
        storage_dir: str,
        config_name: str,
        num_processes: int = 8,
        max_memory: int = 1000_000_000,
        max_shard_size: int = 100_000,
        name_suffix: str = "",
    ):
        self._storage_dir = Path(storage_dir)
        self._config_name = config_name
        self._num_processes = num_processes
        self._max_memory = max_memory
        self._max_shard_size = max_shard_size
        self._name_suffix = name_suffix

        if not self._storage_dir.exists():
            self._storage_dir.mkdir(parents=True, exist_ok=True)
        assert self._storage_dir.is_dir(), (
            f"Storage directory {self._storage_dir} must be a directory."
        )
        (self._storage_dir / config_name).mkdir(parents=True, exist_ok=True)

    def split_dir(self, split: DatasetSplitType) -> Path:
        split_dir = (
            Path(self._storage_dir)
            / f"{self._config_name}/msgpack/{split.value}/{self._name_suffix}"
        )
        split_dir.mkdir(parents=True, exist_ok=True)
        return split_dir

    def dataset_exists(self) -> bool:
        return (Path(self._storage_dir) / f"{self._config_name}/msgpack/").exists()

    def split_exists(self, split: DatasetSplitType) -> bool:
        return len(self.split_files(split)) != 0

    def split_files(self, split: DatasetSplitType) -> bool:
        return list(self.split_dir(split).glob("*.msgpack"))

    def purge_split(self, split: DatasetSplitType) -> None:
        import shutil

        split_dir = self.split_dir(split)
        if split_dir.exists():
            logger.info(
                f"Purging dataset split {split.value} from storage {split_dir}."
            )
            shutil.rmtree(split_dir)

    def write_split(self, split_iterator: SplitIterator) -> None:
        try:
            self._write(split_iterator=split_iterator)
        except Exception as e:
            self.purge_split(split_iterator.split)
            raise RuntimeError(
                f"Error while writing dataset split {split_iterator.split.value} to storage. Cleaning up..."
            ) from e
        except KeyboardInterrupt:
            pass
            # self.purge_split(split_iterator.split)
            # raise KeyboardInterrupt(
            #     "KeyboardInterrupt detected. Stopping dataset preparation..."
            # ) from e

    def map_to_nullable_dtypes(
        self, row_serialization_types: dict[str, type]
    ) -> dict[str, str]:
        type_map = {str: "string", int: "Int64", float: "Float64", bool: "boolean"}
        return {
            col: type_map.get(tp, "object")
            for col, tp in row_serialization_types.items()
        }

    def read_split(
        self,
        split: DatasetSplitType,
        data_model: type[BaseDataInstance],
        output_transform: Callable | None = None,
        allowed_keys: set[str] | None = None,
    ) -> SplitIterator:
        from atria_datasets.core.dataset._datasets import SplitIterator
        from atria_datasets.core.storage.shard_list_datasets import (
            MsgpackShardListDataset,
        )

        if not self.split_exists(split):
            raise RuntimeError(
                f"Dataset split {split.value} not prepared. Please call `write_split()` first."
            )

        if allowed_keys is not None:
            allowed_keys.add("sample_id")
            allowed_keys.add("index")

        return SplitIterator(
            split=split,
            base_iterator=MsgpackShardListDataset(self.split_files(split)),
            input_transform=MsgpackStorageReadTransform(
                data_model=data_model, allowed_keys=allowed_keys
            ),
            output_transform=output_transform,
            data_model=data_model,
        )

    def get_splits(self) -> list[DatasetSplitType]:
        """
        Get a list of available dataset splits in the storage directory.

        Returns:
            list[DatasetSplitType]: A list of available dataset splits.
        """
        from atria_types import DatasetSplitType

        return [split for split in DatasetSplitType if self.split_exists(split)]

    def _write(self, split_iterator: SplitIterator) -> Path:
        """
        Writes a dataset split to storage shards.

        Supports both single-process and multi-process modes using multiprocessing.
        """
        split_dir = self.split_dir(split=split_iterator.split)
        logger.info(
            f"Preprocessing dataset split {split_iterator.split.value} to cached msgpack storage {split_dir}"
        )
        split_iterator.disable_tf()

        if self._num_processes > 1:
            write_infos = write_split_parallel(
                split_iterator=split_iterator,
                split_dir=split_dir,
                max_shard_size=self._max_shard_size,
                num_processes=self._num_processes,
            )
        else:
            write_infos = write_split_single(
                split_iterator=split_iterator,
                split_dir=split_dir,
                max_shard_size=self._max_shard_size,
            )

        # Log the writing results
        total_samples = sum(shard.nsamples for shard in write_infos)
        logger.info(
            f"Successfully wrote {total_samples} samples to {len(write_infos)} shards "
            f"for split {split_iterator.split.value}"
        )

        split_iterator.enable_tf()
        return split_dir
