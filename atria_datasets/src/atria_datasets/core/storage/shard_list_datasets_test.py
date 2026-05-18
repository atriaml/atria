import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from atria_logger import get_logger
from atria_types import DatasetShardInfo
from datadings.reader.msgpack import (
    keys_len,
    legacy_index_len,
    legacy_load_index,
    load_offsets,
    unpackb,
)
from wids import ShardListDataset

logger = get_logger(__name__)


class MsgpackReader:
    """
    Clean msgpack shard reader.

    All metadata loaded eagerly in __init__ (no fd leak).
    Only _infile needs lifecycle management.
    """

    def __init__(self, path: str | Path, buffering: int = 0) -> None:
        self._path = Path(path)
        self._buffering = buffering
        self._infile = None

        if not self._path.exists():
            raise FileNotFoundError(f"{self._path} not found")

        # all of these open, read, close internally — no leak
        try:
            self._len = keys_len(self._path)
        except FileNotFoundError:
            self._len = legacy_index_len(self._path)

        try:
            self._offsets = load_offsets(self._path)
        except FileNotFoundError:
            legacy = legacy_load_index(self._path)
            self._offsets = legacy[1]

    def __len__(self) -> int:
        return self._len

    # -- lifecycle (only _infile) --

    def open(self) -> "MsgpackReader":
        if self._infile is None or self._infile.closed:
            self._infile = open(self._path, "rb", self._buffering)
        return self

    def close(self) -> None:
        if self._infile is not None and not self._infile.closed:
            self._infile.close()
        self._infile = None

    def __enter__(self) -> "MsgpackReader":
        return self.open()

    def __exit__(self, *args) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    # -- read --

    def __getitem__(self, index: int) -> Any:
        return self.get(index)

    def get(self, index: int, raw: bool = False) -> Any:
        if self._infile is None or self._infile.closed:
            raise RuntimeError("Reader not open. Use `with` or call .open()")

        pos = self._offsets
        offset = pos[index]
        n = pos[index + 1] - offset
        self._infile.seek(offset, 0)
        data = self._infile.read(n)

        if not raw:
            data = unpackb(data)
        return data


class MsgpackShardListDataset(Sequence[Any]):
    def __init__(self, shard_files: list[str]) -> None:
        self._shard_files = [str(f) for f in shard_files]
        self._total_size: int = 0

        # no fd opened — MsgpackReader.__init__ only loads metadata
        cumulative_sizes: list[int] = []
        for f in self._shard_files:
            reader = MsgpackReader(f)
            self._total_size += len(reader)
            cumulative_sizes.append(self._total_size)

        self._cumulative_sizes = np.array(cumulative_sizes)

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard_index = int(np.searchsorted(self._cumulative_sizes, index, side="right"))
        if shard_index == 0:
            inner_index = index
        else:
            inner_index = index - int(self._cumulative_sizes[shard_index - 1])

        with MsgpackReader(self._shard_files[shard_index]) as reader:
            sample = reader[inner_index]

        sample.pop("key", None)
        return sample

    def __len__(self) -> int:
        return self._total_size


class TarShardListDataset(ShardListDataset):
    """
    A dataset class for reading tar-based shard files.

    This class provides functionality for loading and iterating over datasets stored
    in tar-based shard files. It supports efficient handling of shard metadata and
    caching for improved performance.

    Attributes:
        cache_dir (Path): The directory used for caching shard data.
    """

    def __init__(self, shard_info_list: list[DatasetShardInfo]) -> None:
        """
        Initializes the `TarShardListDataset`.

        Args:
            shard_files (List[DatasetShardInfo]): A list of shard metadata containing file URLs.
        """
        if isinstance(shard_info_list[0], DatasetShardInfo):
            self.shard_info_list = shard_info_list
            super().__init__(
                [shard_file.model_dump() for shard_file in shard_info_list]
            )
        else:
            import wids

            shard_info_list = [
                {"url": file, "nsamples": wids.wids.compute_num_samples(file)}
                for file in shard_info_list
            ]
            shard_info_list = [
                shard for shard in shard_info_list if shard["nsamples"] > 0
            ]

            super().__init__(shard_info_list)

        # Always clean the cache directory on startup (default is /tmp/wids)
        if Path(self.cache_dir).exists():
            shutil.rmtree(Path(self.cache_dir))
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        # we remove all transformations as we have our own
        self.transformations = []  # type: ignore
