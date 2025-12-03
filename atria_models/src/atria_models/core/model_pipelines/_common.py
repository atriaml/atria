from __future__ import annotations

from typing import TypeVar

from atria_registry import ModuleConfig
from pydantic import BaseModel

from atria_models.core.model_builders._common import FrozenLayers, ModelBuilderType

_DEFAULT_OPTIMIZER_PARAMETERS_KEY = "default"


class ModelConfig(BaseModel):
    # builder type to use for model construction
    builder_type: ModelBuilderType = ModelBuilderType.local
    bn_to_gn: bool = False
    frozen_layers: FrozenLayers | list[str] = FrozenLayers.none
    pretrained_checkpoint: str | None = None

    # Path or name of the model to build
    model_name_or_path: str
    model_kwargs: dict[str, object] = {}


class ModelPipelineConfig(ModuleConfig):
    model_config: ModelConfig


T_ModelPipelineConfig = TypeVar("T_ModelPipelineConfig", bound=ModelPipelineConfig)
