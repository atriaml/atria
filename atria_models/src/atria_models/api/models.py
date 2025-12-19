"""API functions for loading and preprocessing datasets."""

from __future__ import annotations

from atria_logger import get_logger
from atria_types._datasets import DatasetLabels

from atria_models.core.model_pipelines._model_pipeline import ModelPipeline
from atria_models.registry import MODEL_PIPELINES
from atria_models.registry.registry_groups import MODEL

logger = get_logger(__name__)


def load_model_pipeline_config(model_name: str, **kwargs):
    logger.debug(
        f"Loading model pipeline config for model: {model_name} with params: {kwargs}"
    )
    return MODEL_PIPELINES.load_module_config(model_name, **kwargs)


def load_model_pipeline(
    model_pipeline_name: str, labels: DatasetLabels, **kwargs
) -> ModelPipeline:
    config = load_model_pipeline_config(model_pipeline_name, **kwargs)
    return config.build(labels=labels)


def load_model_config(model_name: str, **kwargs):
    logger.debug(f"Loading model config for model: {model_name} with params: {kwargs}")
    return MODEL.load_module_config(model_name, **kwargs)


def load_model(model_name: str, **kwargs):
    config = load_model_config(model_name, **kwargs)
    return config.build()
