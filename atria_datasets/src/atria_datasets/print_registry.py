from atria_logger import get_logger

from atria_datasets import DATASETS

logger = get_logger(__name__)

if __name__ == "__main__":
    logger.info(f"Dataset registry loaded successfully:\n{DATASETS.list_all_modules()}")
