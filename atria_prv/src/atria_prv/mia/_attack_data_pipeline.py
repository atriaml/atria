from __future__ import annotations

import copy
import random
from typing import TYPE_CHECKING

from atria_datasets.core.dataset._partitioning import group_indices_by_original_id
from atria_logger import get_logger
from atria_ml.data_pipeline._utilities import auto_dataloader, default_collate
from realtime import dataclass

if TYPE_CHECKING:
    from atria_datasets.core.dataset._datasets import Dataset
    from atria_datasets.core.dataset._split_iterators import SplitIterator
    from torch.utils.data import DataLoader

logger = get_logger(__name__)


@dataclass
class AttackDataLoaders:
    members_train: DataLoader
    members_test: DataLoader
    non_members_train: DataLoader
    non_members_test: DataLoader


class AttackDataPipeline:
    def __init__(
        self,
        dataset: Dataset,
        *,
        attack_train_ratio: float,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        seed: int = 42,
    ) -> None:
        self._dataset = dataset
        self._attack_train_ratio = attack_train_ratio
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._seed = seed
        self._build()

    def _subset(self, split: SplitIterator, indices: list[int]) -> SplitIterator:
        subset = copy.deepcopy(split)
        subset.subset_indices = indices
        return subset

    def _loader(self, iterator: SplitIterator) -> DataLoader:
        from torch.utils.data import SequentialSampler

        return auto_dataloader(
            dataset=iterator,
            collate_fn=default_collate,
            sampler=SequentialSampler(iterator),
            shuffle=False,
            drop_last=False,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )

    def _create_train_eval_splits(self, dataset: SplitIterator):
        id_to_indices = group_indices_by_original_id(dataset.sample_keys)
        original_ids = sorted(id_to_indices)
        random.Random(self._seed).shuffle(original_ids)

        train_size = int(len(original_ids) * self._attack_train_ratio)
        train_ids, eval_ids = original_ids[:train_size], original_ids[train_size:]

        # Sanity check: no original ID should appear in both splits.
        overlap = set(train_ids) & set(eval_ids)
        if overlap:
            raise RuntimeError(
                f"Train/eval split is not disjoint. Overlapping original IDs: {sorted(overlap)}"
            )

        # assign train indices
        train_indices: list[int] = []
        for original_id in train_ids:
            train_indices.extend(id_to_indices.get(original_id, []))
        train_indices = sorted(train_indices)
        train_dataset = copy.deepcopy(dataset)
        train_dataset.subset_indices = train_indices

        # assign eval indices
        eval_indices: list[int] = []
        for original_id in eval_ids:
            eval_indices.extend(id_to_indices.get(original_id, []))
        eval_indices = sorted(eval_indices)
        eval_dataset = copy.deepcopy(dataset)
        eval_dataset.subset_indices = eval_indices

        return train_dataset, eval_dataset

    def _build(self):
        self._members_dataset, self._non_members_dataset = (
            self._dataset.train,
            self._dataset.test,
        )
        self._members_train, self._members_test = self._create_train_eval_splits(
            self._members_dataset
        )
        self._non_members_train, self._non_members_test = (
            self._create_train_eval_splits(self._non_members_dataset)
        )

    def summarize(self):
        logger.info(
            f"members(train)={len(self._members_dataset)}, non-members(test)={len(self._non_members_dataset)}; "
            f"attack-train={len(self._members_train)} members + {len(self._non_members_train)} non-members, "
            f"attack-test={len(self._members_test)} members + {len(self._non_members_test)} non-members."
        )

    def dataloaders(self):
        return AttackDataLoaders(
            members_train=self._loader(self._members_train),
            non_members_train=self._loader(self._non_members_train),
            members_test=self._loader(self._members_test),
            non_members_test=self._loader(self._non_members_test),
        )
