from atria_logger import get_logger
from atria_transforms.api.tfs import load_transform

from atria_datasets import load_dataset_config

logger = get_logger(__name__)


def main():
    dataset_config = load_dataset_config("due_benchmark/DocVQA")
    dataset = dataset_config.build(
        enable_cached_splits=True,
        preprocess_train_transform=load_transform(
            "unroll_qa_pairs_transform", remove_no_answer_samples=True
        ),
        preprocess_eval_transform=load_transform(
            "unroll_qa_pairs_transform", remove_no_answer_samples=False
        ),
        max_cache_image_size=1024,
        num_processes=8,
    )
    logger.info(f"Loaded dataset:\n{dataset}")

    # get first sample
    sample = next(iter(dataset.train))

    logger.info(f"First sample in train split:\n{sample}")


if __name__ == "__main__":
    main()
