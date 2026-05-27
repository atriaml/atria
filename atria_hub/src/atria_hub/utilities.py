import logging

from atria_hub.config import settings


def get_logger(name: str) -> logging.Logger:
    from atria_core.logger import get_logger

    logger = get_logger(name)

    logging.basicConfig(
        level=logging.INFO, format=settings.LOG_FORMAT, datefmt=settings.DATE_FORMAT
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("lakefs-spec").setLevel(logging.WARNING)

    return logger


def _get_content_type_from_filename(filename: str) -> str:
    import mimetypes

    content_type, _ = mimetypes.guess_type(filename)
    if content_type is None:
        content_type = "application/octet-stream"
    return content_type
