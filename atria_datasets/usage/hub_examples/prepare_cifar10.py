import os

from atria_logger import get_logger

from atria_datasets import load_dataset_config
from atria_datasets.core.dataset._cached_dataset import CachedDataset
from atria_datasets.core.storage.utilities import FileStorageType

logger = get_logger(__name__)


SCRIPT_PATH = os.path.abspath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)


def main():
    dataset_config = load_dataset_config("cifar10/default")
    dataset = dataset_config.build(
        enable_cached_splits=True, cached_storage_type=FileStorageType.DELTALAKE
    )
    logger.info(f"Loaded dataset:\n{dataset}")

    # get first sample
    sample = next(iter(dataset.train))

    logger.info(f"First sample in train split:\n{sample}")

    assert isinstance(dataset, CachedDataset), (
        "Expected dataset to be a CachedDataset after building with caching enabled."
    )
    dataset.upload_to_hub(name="cifar10-example")


if __name__ == "__main__":
    main()
