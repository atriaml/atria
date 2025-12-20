# ruff: noqa

from typing import TYPE_CHECKING

import lazy_loader as lazy

# Ensure registry is initialized immediately
import atria_insights.core.model_pipelines._registry_groups  # noqa: F401

if TYPE_CHECKING:
    from atria_insights.core.model_pipelines._common import (
        ExplainableModelPipelineConfig,
        ExplanationTargetStrategy,
    )
    from atria_insights.core.model_pipelines._image_pipeline import (
        ExplainableImageClassificationPipeline,
    )
    from atria_insights.core.model_pipelines._model_pipeline import (
        ExplainableModelPipeline,
    )
    from atria_insights.core.model_pipelines._registry_groups import (
        EXPLAINABLE_MODEL_PIPELINES,
    )


__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_common": ["ExplainableModelPipelineConfig", "ExplanationTargetStrategy"],
        "_image_pipeline": ["ExplainableImageClassificationPipeline"],
        "_model_pipeline": ["ExplainableModelPipeline"],
        "_registry_groups": ["EXPLAINABLE_MODEL_PIPELINES"],
    },
)
