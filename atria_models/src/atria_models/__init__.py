# ruff: noqa

from typing import TYPE_CHECKING

import lazy_loader as lazy

# Ensure registry is initialized immediately
import atria_models.registry  # noqa: F401

if TYPE_CHECKING:
    from atria_models.api.models import load_model_pipeline, load_model_pipeline_config
    from atria_models.core.model_builders import (
        FrozenLayers,
        ModelBuilder,
        ModelBuilderType,
        TimmModelBuilder,
        TorchvisionModelBuilder,
        TransformersModelBuilder,
    )
    from atria_models.core.model_pipelines import (
        ImageClassificationPipeline,
        ImageModelPipeline,
        ModelPipeline,
        ModelPipelineConfig,
        QuestionAnsweringPipeline,
        SequenceClassificationPipeline,
        SequenceModelPipeline,
        T_ModelPipelineConfig,
        TokenClassificationPipeline,
    )
    from atria_models.registry import MODEL_PIPELINE


__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "api.models": ["load_model_pipeline", "load_model_pipeline_config"],
        "core.model_builders": [
            "FrozenLayers",
            "ModelBuilder",
            "ModelBuilderType",
            "TimmModelBuilder",
            "TorchvisionModelBuilder",
            "TransformersModelBuilder",
        ],
        "core.model_pipelines": [
            "ImageClassificationPipeline",
            "ImageModelPipeline",
            "ModelPipeline",
            "ModelPipelineConfig",
            "QuestionAnsweringPipeline",
            "SequenceClassificationPipeline",
            "SequenceModelPipeline",
            "T_ModelPipelineConfig",
            "TokenClassificationPipeline",
        ],
        "registry": ["MODEL_PIPELINE"],
    },
)
