"""Msgpack-based shard writer implementation."""

from collections.abc import Callable
from typing import Any

from atria_logger import get_logger
from datadings.writer import Writer

logger = get_logger(__name__)


class DuplicateKeyError(Exception):
    pass


class MsgpackFileWriter(Writer):
    def _write_data(self, key, packed):
        if key in self._keys_set:
            raise DuplicateKeyError(f"Duplicate key {key!r} not allowed.")
        self._keys.append(key)
        self._keys_set.add(key)
        self._hash.update(packed)
        self._outfile.write(packed)
        self._offsets.append(self._outfile.tell())
        self.written += 1

    def write(self, sample: dict[str, Any]) -> int:
        assert "key" in sample, "Sample must contain a unique 'key' value."
        self._write(sample["key"], sample)
        return self.written


class MsgpackShardWriter:
    def __init__(
        self,
        pattern: str,
        maxcount: int = 100000,
        maxsize: float = 3e9,
        post: Callable | None = None,
        start_shard: int = 0,
        verbose: int = 0,
        opener: Callable | None = None,
        **kw,
    ):
        self.verbose = verbose
        self.kw = kw
        self.maxcount = maxcount
        self.maxsize = maxsize
        self.post = post

        self.writer = None
        self.shard = start_shard
        self.pattern = pattern
        self.total = 0
        self.count = 0
        self.size = 0
        self.fname = None
        self.opener = opener
        self.next_stream()

    def next_stream(self) -> None:
        self.finish()
        self.fname = self.pattern % self.shard  # type: ignore
        if self.verbose:
            logger.info(
                f"# Writing {self.fname}, {self.count} samples, {self.size / 1e9:.1f} GB, {self.total} total samples."
            )
        self.shard += 1
        if self.opener:
            self.writer = MsgpackFileWriter(
                self.opener(self.fname), **self.kw, disable=True
            )  # type: ignore
        else:
            self.writer = MsgpackFileWriter(self.fname, **self.kw, disable=True)  # type: ignore
        self.count = 0
        self.size = 0

    def write(self, obj: dict[str, Any]) -> None:
        if (
            self.writer is None
            or self.count >= self.maxcount
            or self.size >= self.maxsize
        ):
            self.next_stream()
        size = self.writer.write(obj)  # type: ignore
        self.count += 1
        self.total += 1
        self.size += size

    def finish(self) -> None:
        if self.writer is not None:
            self.writer.close()
            assert self.fname is not None
            if callable(self.post):
                self.post(self.fname)
            self.writer = None

    def close(self) -> None:
        self.finish()
        del self.writer
        del self.shard
        del self.count
        del self.size

    def __enter__(self):
        return self

    def __exit__(self, *args, **kw):
        self.close()
