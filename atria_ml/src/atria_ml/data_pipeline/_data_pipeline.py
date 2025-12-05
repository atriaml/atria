from __future__ import annotations

from typing import TYPE_CHECKING

from atria_datasets.core.dataset._datasets import Dataset
from atria_datasets.core.dataset._split_iterators import SplitIterator
from atria_logger import get_logger
from atria_types._utilities._repr import RepresentationMixin

from atria_ml.data_pipeline._utilities import auto_dataloader, default_collate

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


logger = get_logger(__name__)


class DataPipeline(RepresentationMixin):
    def __init__(self, dataset: Dataset, collate_fn: str | None = "default_collate"):
        self._dataset = dataset
        self._sharded_storage_kwargs = {}
        self._dataset_splitter = None

        if collate_fn == "default_collate":
            self._collate_fn = default_collate
        elif collate_fn == "identity":
            self._collate_fn = lambda x: x
        else:
            raise ValueError(f"Invalid collate_fn: {collate_fn}")

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

    def train_dataloader(
        self,
        batch_size: int = 1,
        pin_memory: bool = True,
        num_workers: int = 4,
        shuffle: bool = True,
    ) -> DataLoader:
        import ignite.distributed as idist
        from torch.utils.data import RandomSampler, SequentialSampler

        if self._dataset.train is None:
            raise ValueError("No training dataset found.")

        return auto_dataloader(
            dataset=self._dataset.train,
            collate_fn=self._collate_fn,
            sampler=RandomSampler(self._dataset.train)
            if shuffle
            else SequentialSampler(self._dataset.train),
            drop_last=idist.get_world_size() > 1,
            batch_size=batch_size * idist.get_world_size(),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def validation_dataloader(
        self, batch_size: int = 1, pin_memory: bool = True, num_workers: int = 4
    ) -> DataLoader:
        dataset = self._dataset.validation or self._dataset.test
        if dataset is None:
            raise ValueError("No validation or test dataset found.")

        if self._dataset.validation is None:
            logger.warning(
                "No validation dataset found, using test dataset for validation."
            )

        return self._build_evaluation_dataloader(
            dataset,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )

    def test_dataloader(
        self, batch_size: int = 1, pin_memory: bool = True, num_workers: int = 4
    ) -> DataLoader:
        if self._dataset.test is None:
            raise ValueError("No test dataset found.")
        return self._build_evaluation_dataloader(
            self._dataset.test,
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
            collate_fn=self._collate_fn,
            shuffle=False,
            drop_last=False,
            sampler=SequentialSampler(dataset),
            batch_size=batch_size * idist.get_world_size(),
            pin_memory=pin_memory,
            num_workers=num_workers,
        )
