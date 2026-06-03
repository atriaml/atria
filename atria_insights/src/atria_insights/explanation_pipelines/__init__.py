# ruff: noqa

from typing import TYPE_CHECKING

import lazy_loader as lazy

# Ensure registry is initialized immediately
import atria_insights.explanation_pipelines._registry_groups  # noqa: F401

if TYPE_CHECKING:
    from atria_insights.explanation_pipelines._common import (
        ExplanationPipelineConfig,
        ExplanationTargetStrategy,
    )
    from atria_insights.explanation_pipelines._image_pipeline import (
        ImageClassificationExplanationPipeline,
    )
    from atria_insights.explanation_pipelines._model_pipeline import ExplanationPipeline
    from atria_insights.explanation_pipelines._registry_groups import (
        EXPLANATION_PIPELINES,
    )


__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_common": ["ExplanationPipelineConfig", "ExplanationTargetStrategy"],
        "_image_pipeline": ["ImageClassificationExplanationPipeline"],
        "_model_pipeline": ["ExplanationPipeline"],
        "_registry_groups": ["EXPLANATION_PIPELINES"],
    },
)
