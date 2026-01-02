from __future__ import annotations

from collections import OrderedDict
from typing import TypeVar

import torch
from atria_logger import get_logger
from atria_models.core.model_pipelines._image_pipeline import (
    ImageClassificationPipelineConfig,
    ImageModelPipeline,
    ImageModelPipelineConfig,
)
from atria_transforms.data_types._document import DocumentTensorDataModel
from atria_transforms.data_types._image import ImageTensorDataModel
from atria_types._datasets import DatasetLabels

from atria_insights.baseline_generators._simple import SimpleBaselineGeneratorConfig
from atria_insights.robustness_eval_pipelines._common import (
    RobustnessEvalModelPipelineConfig,
)
from atria_insights.robustness_eval_pipelines._model_pipeline import (
    RobustnessEvalModelPipeline,
)

logger = get_logger(__name__)


class RobustnessEvalImageModelPipelineConfig(RobustnessEvalModelPipelineConfig):
    model_pipeline: ImageModelPipelineConfig
    baseline_generator: SimpleBaselineGeneratorConfig = SimpleBaselineGeneratorConfig()


T_RobustnessEvalImageModelPipelineConfig = TypeVar(
    "T_RobustnessEvalImageModelPipelineConfig",
    bound="RobustnessEvalImageModelPipelineConfig",
)


class RobustnessEvalImageModelPipeline(
    RobustnessEvalModelPipeline[
        RobustnessEvalImageModelPipelineConfig,
        ImageTensorDataModel | DocumentTensorDataModel,
    ]
):
    __abstract__ = True
    __config__ = RobustnessEvalImageModelPipelineConfig

    def __init__(
        self,
        config: RobustnessEvalImageModelPipelineConfig,
        labels: DatasetLabels,
        persist_to_disk: bool = True,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__(
            config=config,
            labels=labels,
            persist_to_disk=persist_to_disk,
            cache_dir=cache_dir,
        )
        assert isinstance(self._model_pipeline, ImageModelPipeline), (
            f"{self.__class__.__name__} can only be used with ImageModelPipeline. Found {self._model_pipeline=}"
        )

    def _features(
        self, batch: ImageTensorDataModel | DocumentTensorDataModel
    ) -> torch.Tensor | OrderedDict[str, torch.Tensor]:
        return OrderedDict(image=batch.images)


class RobustnessEvalImageClassificationPipelineConfig(
    RobustnessEvalImageModelPipelineConfig
):
    model_pipeline: ImageClassificationPipelineConfig = (
        ImageClassificationPipelineConfig()
    )

    @property
    def name(self) -> str:
        return "image_classification"


class RobustnessEvalImageClassificationPipeline(RobustnessEvalImageModelPipeline):
    __config__ = RobustnessEvalImageClassificationPipelineConfig
