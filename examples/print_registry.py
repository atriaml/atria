from atria_datasets import DATASETS
from atria_logger import get_logger
from atria_metrics import METRICS
from atria_models import MODEL_PIPELINES, MODELS
from atria_transforms import DATA_TRANSFORMS

logger = get_logger(__name__)

if __name__ == "__main__":
    logger.info("Model registry loaded successfully:")
    print("DB Path", MODEL_PIPELINES._db_path())
    logger.info(f"Model pipelines:\n{MODEL_PIPELINES.list_all_modules()}")
    logger.info(f"Models:\n{MODELS.list_all_modules()}")

    logger.info("Dataset registry loaded successfully:")
    print("DB Path", DATASETS._db_path())
    logger.info(f"Datasets:\n{DATASETS.list_all_modules()}")

    logger.info("Metric registry loaded successfully:")
    logger.info(f"Metrics:\n{METRICS.list_all_modules()}")

    logger.info("Transform registry loaded successfully:")
    logger.info(f"Transforms:\n{DATA_TRANSFORMS.list_all_modules()}")
