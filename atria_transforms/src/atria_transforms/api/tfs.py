"""API functions for loading and preprocessing datasets."""

from __future__ import annotations

from atria_logger import get_logger
from atria_transforms.core import DataTransform
from atria_transforms.registry import DATA_TRANSFORM

logger = get_logger(__name__)


def load_transform(transform_name: str, **kwargs) -> DataTransform:
    logger.debug(f"Loading transform: {transform_name} with args: {kwargs}")
    return DATA_TRANSFORM.load_module_config(transform_name, **kwargs)
