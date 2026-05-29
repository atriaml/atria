from atria_logger import get_logger
from atria_transforms.api.tfs import load_transform

from atria_datasets import load_dataset_config

logger = get_logger(__name__)


def main():
    dataset_config = load_dataset_config("due_benchmark/WikiTableQuestions", max_train_samples=100, max_validation_samples=100, max_test_samples=100)
    dataset = dataset_config.build(enable_cached_splits=True, num_processes=4)
    logger.info(f"Loaded dataset:\n{dataset}")

    # get first sample
    sample = next(iter(dataset.train))

    logger.info(f"First sample in train split:\n{sample}")


if __name__ == "__main__":
    main()
