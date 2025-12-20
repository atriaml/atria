from atria_logger import get_logger
from atria_transforms.api.tfs import load_transform

from atria_datasets import load_dataset_config

logger = get_logger(__name__)


def main():
    dataset_config = load_dataset_config("due_benchmark/DocVQA")
    dataset = dataset_config.build(enable_cached_splits=True)
    logger.info(f"Loaded dataset:\n{dataset}")

    # get first sample
    sample = next(iter(dataset.train))

    logger.info(f"First sample in train split:\n{sample}")

    # process the dataset with a custom transform
    processed_dataset = dataset.process_dataset(
        train_transform=load_transform(
            "unroll_qa_pairs_transform", remove_no_answer_samples=True
        ),
        eval_transform=load_transform(
            "unroll_qa_pairs_transform", remove_no_answer_samples=False
        ),
        max_cache_image_size=512,
        num_processes=8,
    )

    # get first sample after processing
    logger.info(processed_dataset)

    for sample in processed_dataset.train:
        logger.info(f"Processed sample in train split:\n{sample}")
        break


if __name__ == "__main__":
    main()
