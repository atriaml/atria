from __future__ import annotations

from typing import TYPE_CHECKING, Generic

from atria_logger import get_logger

from atria_models.core.model_pipelines._common import T_ModelPipelineConfig
from atria_models.core.model_pipelines._model_pipeline import ModelPipeline

if TYPE_CHECKING:
    pass


logger = get_logger(__name__)


class StateDictHandler(Generic[T_ModelPipelineConfig]):
    def __init__(self, model_pipeline: ModelPipeline[T_ModelPipelineConfig]) -> None:
        self._model_pipeline = model_pipeline

    def state_dict(self) -> dict:
        return {
            "model_pipeline_config": self._model_pipeline.config.model_dump(),
            "model": self._model_pipeline._model.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if "model_pipeline_config" in state_dict:
            self._model_pipeline.config.model_validate(
                state_dict["model_pipeline_config"]
            )
        if "model" in state_dict:
            self._model_pipeline._model.load_state_dict(
                state_dict["model"], strict=True
            )
