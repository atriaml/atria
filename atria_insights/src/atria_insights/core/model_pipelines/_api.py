"""API functions for loading and preprocessing datasets."""

from __future__ import annotations

from atria_logger import get_logger
from atria_models.core.model_pipelines._model_pipeline import ModelPipeline
from atria_types._datasets import DatasetLabels

from atria_insights.core.model_pipelines._registry_groups import (
    EXPLAINABLE_MODEL_PIPELINES,
)

logger = get_logger(__name__)


def load_x_model_pipeline_config(model_name: str, **kwargs):
    logger.debug(
        f"Loading model pipeline config for model: {model_name} with params: {kwargs}"
    )
    return EXPLAINABLE_MODEL_PIPELINES.load_module_config(model_name, **kwargs)


def load_x_model_pipeline(
    model_name: str, labels: DatasetLabels, **kwargs
) -> ModelPipeline:
    logger.debug(
        f"Loading model pipeline for model: {model_name} with labels: {labels} and params: {kwargs}"
    )
    config = load_x_model_pipeline_config(model_name, **kwargs)
    return config.build(labels=labels)
