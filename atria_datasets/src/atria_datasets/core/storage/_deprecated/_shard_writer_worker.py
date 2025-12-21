import os
from pathlib import Path

from atria_types._datasets import DatasetShardInfo
from ray import logger

from atria_datasets.core.storage._mp._shard_writer_worker import ShardWriterWorker
from atria_datasets.core.storage._shard_writers._msgpack import DuplicateKeyError
from atria_datasets.core.storage.utilities import FileStorageType

_writer: ShardWriterWorker | None = None


def init_writer(split_dir: Path, data_model, max_shard_size: int, preprocess_transform):
    global _writer

    pid = os.getpid()
    shard_file_pattern = str(split_dir / f"{pid}-%06d.msgpack")

    _writer = ShardWriterWorker(
        data_model=data_model,
        storage_type=FileStorageType.MSGPACK,
        storage_file_pattern=shard_file_pattern,
        max_shard_size=max_shard_size,
        preprocess_transform=preprocess_transform,
    )
    _writer.load()


def write_one(item):
    assert _writer is not None, "Writer not initialized."
    idx, sample = item
    try:
        _writer.write(idx, sample)
    except DuplicateKeyError:
        logger.warning(
            f"Duplicate key detected at sample index {idx}. Skipping sample."
        )
    except Exception as e:
        raise e
    return None


def finalize_worker(_) -> list[DatasetShardInfo]:
    global _writer
    if _writer is None:
        return []
    info = _writer.close()
    _writer = None
    return info
