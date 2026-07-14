from __future__ import annotations

from typing import TYPE_CHECKING

from atria_models.core.model_pipelines import TokenClassificationPipeline

from atria_prv.att._features._bio_scheme import BioScheme
from atria_prv.att._features._extractor import AggConfig, TokenSignalExtractor
from atria_prv.att._features._signals import SIGNAL_FUNCS

if TYPE_CHECKING:
    from atria_models.core.model_pipelines._model_pipeline import ModelPipeline


def build_feature_extractor(model_pipeline: ModelPipeline) -> TokenSignalExtractor:
    """Select and build the feature extractor for a given model pipeline's type."""
    if isinstance(model_pipeline, TokenClassificationPipeline):
        assert model_pipeline._labels.ser is not None, (
            "Labels must be provided for ser tasks."
        )

        return TokenSignalExtractor(
            signals=SIGNAL_FUNCS,
            bio_scheme=BioScheme.from_label_names(model_pipeline._labels.ser),
            config=AggConfig(),
            num_labels=len(model_pipeline._labels.ser),
        )
    raise ValueError(
        f"Unsupported model pipeline type '{type(model_pipeline).__name__}' for "
        f"feature extraction. Supported: TokenClassificationPipeline."
    )
