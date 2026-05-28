
from atria_logger import get_logger
from atria_models import MODEL_PIPELINES, MODELS

logger = get_logger(__name__)

if __name__ == "__main__":
    logger.info("Model registry loaded successfully:")
    logger.info(f"Model pipelines:\n{MODEL_PIPELINES.list_all_modules()}")
    logger.info(f"Models:\n{MODELS.list_all_modules()}")