import hashlib
import json
import random
import re
from pathlib import Path
from typing import Self

from atria_transforms.core._tfs._base import DataTransform
from atria_types import DatasetSplitType

from atria_datasets.core.dataset._cached_dataset import CachedDataset
from atria_datasets.core.dataset._exceptions import SplitNotFoundError

_OVERFLOW_KEY_RE = re.compile(r"^(?P<original_id>.+)_overflow_(?P<overflow_idx>\d+)$")


def resolve_original_id(sample_key: str) -> str:
    """Strips the `_overflow_<n>` suffix added by explode_instances, if present."""
    match = _OVERFLOW_KEY_RE.match(sample_key)
    return match.group("original_id") if match else sample_key


def group_indices_by_original_id(sample_keys: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, key in enumerate(sample_keys):
        groups.setdefault(resolve_original_id(key), []).append(index)
    return groups


class PartitionedDataset(CachedDataset):
    def __init__(
        self,
        path: Path | str,
        partition_id: int,
        partition_cache_dir: str,
        allowed_keys: set[str] | None = None,
        train_transform: DataTransform | None = None,
        eval_transform: DataTransform | None = None,
    ) -> None:
        super().__init__(
            path=path,
            allowed_keys=allowed_keys,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )
        self._partition_id = partition_id
        self._cache_dir = Path(path) / partition_cache_dir

    def load(self) -> Self:
        super().load()
        for split, indices in self._load_partition_indices().items():
            if split not in self._split_iterators:
                raise SplitNotFoundError(
                    f"Split {split.value} not found in dataset at {self._path}"
                )
            self._split_iterators[split].subset_indices = indices
        return self

    def _cache_path(self, split: DatasetSplitType) -> Path:
        return self._cache_dir / f"{split.value}.json"

    def _load_partition_indices(self) -> dict[DatasetSplitType, list[int]]:
        partition_indices: dict[DatasetSplitType, list[int]] = {}
        for split, split_iterator in self._split_iterators.items():
            cache_path = self._cache_path(split)
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"Partition cache not found at {cache_path}. "
                    f"Run DatasetPartitioner(...).compute() first."
                )
            all_partitions: list[list[str]] = json.loads(cache_path.read_text())
            if self._partition_id >= len(all_partitions):
                raise ValueError(
                    f"partition_id={self._partition_id} out of range "
                    f"(cache has {len(all_partitions)} partitions)."
                )
            id_to_indices = group_indices_by_original_id(split_iterator.sample_keys)
            indices: list[int] = []
            for original_id in all_partitions[self._partition_id]:
                indices.extend(id_to_indices.get(original_id, []))
            partition_indices[split] = sorted(indices)
        return partition_indices


class DatasetPartitioner:
    def __init__(self, dataset: CachedDataset, n_partitions: int, seed: int = 42):
        assert n_partitions > 0, "n_partitions must be positive"
        self._dataset = dataset
        self._n_partitions = n_partitions
        self._seed = seed
        self._cache_dir = Path(dataset.data_dir) / self.partition_cache_dir

    @property
    def partition_cache_dir(self) -> str:
        config = {"n_partitions": self._n_partitions, "seed": self._seed}
        return (
            "partition_"
            + hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
        )

    def _cache_path(self, split: DatasetSplitType) -> Path:
        return self._cache_dir / f"{split.value}.json"

    def compute(self, force: bool = False) -> None:
        for split, split_iterator in self._dataset.split_iterators.items():
            cache_path = self._cache_path(split)
            if cache_path.exists() and not force:
                continue
            id_to_indices = group_indices_by_original_id(split_iterator.sample_keys)
            original_ids = sorted(id_to_indices)
            random.Random(self._seed).shuffle(original_ids)
            id_groups = [
                original_ids[i :: self._n_partitions] for i in range(self._n_partitions)
            ]
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(id_groups))

    def partition_sizes(self) -> dict[DatasetSplitType, list[int]]:
        return {
            split: [len(g) for g in json.loads(self._cache_path(split).read_text())]
            for split in self._dataset.split_iterators
        }
