from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from atria_models import ModelPipelineConfig

if TYPE_CHECKING:
    pass


class XAIModelPipelineConfig(ModelPipelineConfig):
    pass


T_XAIModelPipelineConfig = TypeVar(
    "T_XAIModelPipelineConfig", bound=XAIModelPipelineConfig
)
