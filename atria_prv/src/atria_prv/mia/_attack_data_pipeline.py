from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from atria_datasets.core.dataset._partitioning import group_indices_by_original_id
from atria_logger import get_logger
from atria_ml.data_pipeline._utilities import auto_dataloader, default_collate

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
        balanced: bool = True,
        seed: int = 42,
    ) -> None:
        self._dataset = dataset
        self._attack_train_ratio = attack_train_ratio
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._balanced = balanced
        self._seed = seed
        self._build()

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

    def _shuffled_original_ids(self, dataset: SplitIterator) -> list[str]:
        ids = sorted(group_indices_by_original_id(dataset.sample_keys))
        random.Random(self._seed).shuffle(ids)
        return ids

    def _create_train_eval_splits(
        self, dataset: SplitIterator, original_ids: list[str]
    ):
        """Split a pre-selected list of original ids into attack-train / attack-eval."""
        id_to_indices = group_indices_by_original_id(dataset.sample_keys)

        train_size = int(len(original_ids) * self._attack_train_ratio)
        train_ids, eval_ids = original_ids[:train_size], original_ids[train_size:]

        def _subset_for(split: SplitIterator, ids: list[str]) -> SplitIterator:
            indices = sorted(i for oid in ids for i in id_to_indices.get(oid, []))
            subset = copy.deepcopy(split)
            subset.subset_indices = indices
            return subset

        return _subset_for(dataset, train_ids), _subset_for(dataset, eval_ids)

    def _build(self):
        self._members_dataset = self._dataset.train
        self._non_members_dataset = self._dataset.test

        member_ids = self._shuffled_original_ids(self._members_dataset)
        non_member_ids = self._shuffled_original_ids(self._non_members_dataset)

        self._members_train, self._members_test = self._create_train_eval_splits(
            self._members_dataset, member_ids
        )
        self._non_members_train, self._non_members_test = (
            self._create_train_eval_splits(self._non_members_dataset, non_member_ids)
        )

        if self._balanced:
            k = min(len(self._members_test), len(self._non_members_test))
            self._members_test.subset_indices = self._members_test.subset_indices[:k]
            self._non_members_test.subset_indices = (
                self._non_members_test.subset_indices[:k]
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
