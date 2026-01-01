from __future__ import annotations

from typing import TYPE_CHECKING

from atria_datasets.core.dataset._datasets import Dataset
from atria_datasets.core.dataset._split_iterators import SplitIterator
from atria_logger import get_logger
from atria_types._common import DatasetSplitType
from atria_types._utilities._repr import RepresentationMixin

from atria_ml.data_pipeline._utilities import auto_dataloader, default_collate

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


logger = get_logger(__name__)


class DataPipeline(RepresentationMixin):
    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset
        self._sharded_storage_kwargs = {}
        self._dataset_splitter = None

    @property
    def dataset(self):
        return self._dataset

    @property
    def dataset_metadata(self):
        return self._dataset.metadata

    def dataloader(
        self,
        split: str,
        batch_size: int = 1,
        pin_memory: bool = True,
        num_workers: int = 4,
        shuffle: bool = True,
    ):
        if split == "train":
            return self.train_dataloader(
                batch_size=batch_size,
                pin_memory=pin_memory,
                num_workers=num_workers,
                shuffle=shuffle,
            )
        elif split == "validation":
            return self.validation_dataloader(
                batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers
            )
        elif split == "test":
            return self.test_dataloader(
                batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers
            )
        else:
            raise ValueError(f"Invalid split name: {split}")

    def get_split_subset(self, dataset: SplitIterator, subset_size: int | None):
        if subset_size is not None:
            original_size = len(self._dataset.test)
            if subset_size > original_size:
                logger.warning(
                    f"Requested eval subset size {subset_size} is larger than "
                    f"the dataset size {original_size}. Using full dataset instead."
                )
                subset_size = original_size
            dataset = self._dataset.test.get_random_subset(subset_size=subset_size)
            assert len(dataset) == subset_size, (
                "Something went wrong when creating the subset."
            )
            logger.info(
                f"Using a subset of the test dataset for evaluation: "
                f"{subset_size} / {original_size} samples."
            )
        return dataset

    def train_dataloader(
        self,
        batch_size: int = 1,
        pin_memory: bool = True,
        num_workers: int = 4,
        shuffle: bool = True,
        subset_size: int | None = None,
    ) -> DataLoader:
        import ignite.distributed as idist
        from torch.utils.data import RandomSampler, SequentialSampler

        dataset = self._dataset.train
        if subset_size is not None:
            dataset = self.get_split_subset(dataset, subset_size=subset_size)

        return auto_dataloader(
            dataset=dataset,
            collate_fn=default_collate,
            sampler=RandomSampler(self._dataset.train)
            if shuffle
            else SequentialSampler(self._dataset.train),
            drop_last=idist.get_world_size() > 1,
            batch_size=batch_size * idist.get_world_size(),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def validation_dataloader(
        self,
        batch_size: int = 1,
        pin_memory: bool = True,
        num_workers: int = 4,
        subset_size: int | None = None,
    ) -> DataLoader:
        if self._dataset.split_exists(DatasetSplitType.validation):
            dataset = self._dataset.validation
        elif self._dataset.split_exists(DatasetSplitType.test):
            dataset = self._dataset.test
            logger.warning(
                "No validation dataset found, using test dataset for validation."
            )
        else:
            raise ValueError("No test or validation dataset found.")

        if subset_size is not None:
            dataset = self.get_split_subset(dataset, subset_size=subset_size)

        return self._build_evaluation_dataloader(
            dataset,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )

    def test_dataloader(
        self,
        batch_size: int = 1,
        pin_memory: bool = True,
        num_workers: int = 4,
        subset_size: int | None = None,
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

        if subset_size is not None:
            dataset = self.get_split_subset(dataset, subset_size=subset_size)

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
        )
