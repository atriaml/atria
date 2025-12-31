import os

from atria_logger import get_logger
from atria_transforms.tfs._document_processor._task_tfs import (
    SequenceClassificationDocumentProcessor,
)
from atria_transforms.tfs._document_tokenizer import DocumentTokenizer

from atria_datasets import load_dataset_config

logger = get_logger(__name__)

SCRIPT_PATH = os.path.abspath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)


def main():
    dataset_config = load_dataset_config("tobacco3482/image_with_ocr")
    dataset = dataset_config.build(enable_cached_splits=True, max_cache_image_size=224)
    logger.info(f"Loaded dataset:\n{dataset}")

    # get first sample
    sample = next(iter(dataset.train))

    logger.info(f"First sample in train split:\n{sample}")

    # process the dataset with a custom transform
    processed_dataset = dataset.process_dataset(
        train_transform=DocumentTokenizer(image_size=(224, 224)),
        eval_transform=DocumentTokenizer(image_size=(224, 224)),
        num_processes=8,
    )
    logger.info(f"Processed dataset:\n{processed_dataset}")

    for sample in processed_dataset.train:
        logger.info(f"First processed sample in train split:\n{sample}")
        break

    tf = SequenceClassificationDocumentProcessor()

    processed_dataset.train.output_transform = tf
    for sample in processed_dataset.train:
        logger.info(f"First tokenized sample in train split:\n{sample}")
        break


if __name__ == "__main__":
    main()
