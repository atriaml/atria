from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from atria_datasets.core.dataset._partitioning import group_indices_by_original_id
from atria_logger import get_logger
from atria_ml.data_pipeline._utilities import auto_dataloader, default_collate
from atria_types._common import DatasetSplitType

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
        seed_train_val: int = 42,
        seed_train_reshuffle: int = 0,
    ) -> None:
        self._dataset = dataset
        self._attack_train_ratio = attack_train_ratio
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._balanced = balanced
        self._seed_train_val = seed_train_val
        self._seed_train_reshuffle = seed_train_reshuffle
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
        random.Random(self._seed_train_val).shuffle(ids)
        return ids

    @staticmethod
    def _subset_by_ids(split: SplitIterator, ids: list[str]) -> SplitIterator:
        """Deep-copy ``split`` and restrict it to the samples of the given original ids."""
        id_to_indices = group_indices_by_original_id(split.sample_keys)
        indices = [i for oid in ids for i in id_to_indices.get(oid, [])]
        subset = copy.deepcopy(split)
        subset.subset_indices = indices
        return subset

    def _create_train_eval_splits(self, dataset: SplitIterator):
        """Split a pre-selected list of original ids into attack-train / attack-eval."""
        original_ids = self._shuffled_original_ids(dataset)

        train_size = int(len(original_ids) * self._attack_train_ratio)
        train_ids, eval_ids = original_ids[:train_size], original_ids[train_size:]

        if self._seed_train_reshuffle:
            random.Random(self._seed_train_reshuffle).shuffle(train_ids)

        logger.info(f"Train ids: {train_ids[:10]}")
        logger.info(f"Test ids: {eval_ids[:10]}")

        return (
            self._subset_by_ids(dataset, train_ids),
            self._subset_by_ids(dataset, eval_ids),
        )

    # ------------------------------------------------------------------ shadow mode
    def shadow_splits(
        self, *, num_models: int, in_ratio: float, seed: int
    ) -> list[tuple[SplitIterator, SplitIterator]]:
        """Per-shadow (in, out) splits of the target train set.

        For each shadow model the train original-ids are randomly split into an ``in``
        (member) half and a disjoint ``out`` (non-member) half, seeded per shadow so the
        splits differ. The ``in`` split trains the shadow; both are queried to build the
        attack-training features. The target test split is left untouched (it is reserved
        as target non-members for evaluation).
        """
        train = self._dataset.train
        ids = sorted(group_indices_by_original_id(train.sample_keys))

        splits: list[tuple[SplitIterator, SplitIterator]] = []
        for i in range(num_models):
            shuffled = list(ids)
            random.Random(seed + i).shuffle(shuffled)
            n_in = int(len(shuffled) * in_ratio)
            in_ids, out_ids = shuffled[:n_in], shuffled[n_in:]
            splits.append(
                (
                    self._subset_by_ids(train, in_ids),
                    self._subset_by_ids(train, out_ids),
                )
            )
        return splits

    def target_eval_splits(self) -> tuple[SplitIterator, SplitIterator]:
        """Full target members (train) and non-members (test), balanced-truncated.

        Used in shadow mode where none of the target data trains the attack model, so the
        whole train/test pools can be used for evaluation (no ``attack_train_ratio`` split).
        """
        members = copy.deepcopy(self._dataset.train)
        non_members = copy.deepcopy(self._dataset.test)
        if self._balanced:
            k = min(len(members), len(non_members))
            members.subset_indices = list(range(len(members)))[:k]
            non_members.subset_indices = list(range(len(non_members)))[:k]
        return members, non_members

    def loader(self, iterator: SplitIterator) -> DataLoader:
        """Public accessor for a sequential eval dataloader over ``iterator``."""
        return self._loader(iterator)

    def _build(self):
        self._members_train, self._members_test = self._create_train_eval_splits(
            self._dataset.train
        )
        self._non_members_train, self._non_members_test = (
            self._create_train_eval_splits(self._dataset.test)
        )

        if self._balanced:
            k = min(len(self._members_test), len(self._non_members_test))

            self._members_train.subset_indices = self._members_train.subset_indices[:k]
            self._non_members_train.subset_indices = (
                self._non_members_train.subset_indices[:k]
            )

            self._members_test.subset_indices = self._members_test.subset_indices[:k]
            self._non_members_test.subset_indices = (
                self._non_members_test.subset_indices[:k]
            )

            logger.info(
                "Member train subset ids: %s", self._members_train.subset_indices[:10]
            )
            logger.info(
                "Non-member train subset ids: %s",
                self._non_members_train.subset_indices[:10],
            )
            logger.info(
                "Member test subset ids: %s", self._members_test.subset_indices[:10]
            )
            logger.info(
                "Non-member test subset ids: %s",
                self._non_members_test.subset_indices[:10],
            )

            # for sanity check see subset ids dont overlap
            set1 = set(self._members_train.subset_indices).intersection(
                self._members_test.subset_indices
            )
            set2 = set(self._non_members_train.subset_indices).intersection(
                self._non_members_test.subset_indices
            )
            assert len(set1) == 0 and len(set2) == 0, (
                "The member and non-member train and test sets must not overlap."
            )

    def summarize(self):
        logger.info(
            f"members(train)={len(self._dataset.train)}, non-members(test)={len(self._dataset.test)}; "
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

    def test_dataloader(
        self, batch_size: int = 1, pin_memory: bool = True, num_workers: int = 4
    ) -> DataLoader:
        if self._dataset.split_exists(DatasetSplitType.test):
            dataset = self._dataset.test
        elif self._dataset.split_exists(DatasetSplitType.validation):
            dataset = self._dataset.validation
            logger.warning(
                "No test dataset found, using validation dataset for testing."
            )
        else:
            raise ValueError("No test or validation dataset found.")

        return self._build_evaluation_dataloader(
            dataset,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )

    def _build_evaluation_dataloader(
        self,
        dataset: SplitIterator,
        batch_size: int = 1,
        pin_memory: bool = True,
        num_workers: int = 4,
    ) -> DataLoader:
        if dataset is None:
            return None

        import ignite.distributed as idist  # type: ignore
        from torch.utils.data import SequentialSampler  # type: ignore

        if idist.get_world_size() > 1:
            if len(dataset) % idist.get_world_size() != 0:
                logger.warning(
                    "Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
        return auto_dataloader(
            dataset=dataset,
            collate_fn=default_collate,
            shuffle=False,
            drop_last=False,
            sampler=SequentialSampler(dataset),
            batch_size=batch_size * idist.get_world_size(),
            pin_memory=pin_memory,
            num_workers=num_workers,
            # persistent_workers=True,
        )
