"""API functions for loading and preprocessing datasets."""

from __future__ import annotations

from atria_logger import get_logger

from atria_models.core.model_pipelines import ModelPipeline, ModelPipelineConfig
from atria_models.registry import MODEL_PIPELINE

logger = get_logger(__name__)


def load_model_pipeline(model_name: str, **kwargs) -> ModelPipeline:
    return MODEL_PIPELINE.load_module(model_name, **kwargs)


def load_model_pipeline_config(model_name: str, **kwargs) -> ModelPipelineConfig:
    _, config = MODEL_PIPELINE.load_module_config(model_name, **kwargs)
    return config
