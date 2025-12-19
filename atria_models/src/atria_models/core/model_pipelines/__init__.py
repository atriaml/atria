# ruff: noqa

from typing import TYPE_CHECKING

import lazy_loader as lazy

if TYPE_CHECKING:
    from ._common import ModelConfig, ModelPipelineConfig, T_ModelPipelineConfig
    from ._image_pipeline import ImageClassificationPipeline, ImageModelPipeline
    from ._model_pipeline import ModelPipeline
    from ._sequence_pipeline import (
        LayoutTokenClassificationPipeline,
        QuestionAnsweringPipeline,
        SequenceClassificationPipeline,
        SequenceModelPipeline,
        TokenClassificationPipeline,
    )

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_model_pipeline": ["ModelPipeline"],
        "_image_pipeline": ["ImageModelPipeline", "ImageClassificationPipeline"],
        "_sequence_pipeline": [
            "SequenceModelPipeline",
            "SequenceClassificationPipeline",
            "TokenClassificationPipeline",
            "LayoutTokenClassificationPipeline",
            "QuestionAnsweringPipeline",
        ],
        "_common": ["ModelPipelineConfig", "T_ModelPipelineConfig", "ModelConfig"],
    },
)
