from __future__ import annotations

from typing import TYPE_CHECKING

from atria_prv.att._features._bio_scheme import BioScheme
from atria_prv.att._features._extractor import AggConfig, TokenSignalExtractor
from atria_prv.att._features._signals import SIGNAL_FUNCS

if TYPE_CHECKING:
    from atria_models.core.model_pipelines._model_pipeline import ModelPipeline
    from atria_types._datasets import DatasetLabels

_SUPPORTED_PIPELINE_NAMES = ("token_classification",)


def build_feature_extractor(
    model_pipeline: ModelPipeline, labels: DatasetLabels
) -> TokenSignalExtractor:
    """Select and build the feature extractor for a given model pipeline's type."""
    pipeline_name = model_pipeline.__pipeline_name__
    if pipeline_name == "token_classification":
        return TokenSignalExtractor(
            signals=SIGNAL_FUNCS,
            bio_scheme=BioScheme.from_label_names(labels.ser),
            config=AggConfig(),
            num_labels=len(labels.ser),
        )
    raise ValueError(
        f"Unsupported model pipeline type '{pipeline_name}' for feature extraction. "
        f"Supported pipeline types: {list(_SUPPORTED_PIPELINE_NAMES)}."
    )
