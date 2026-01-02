from typing import Annotated

from pydantic import Field

from atria_insights.feature_segmentors._base import NoOpSegmenterConfig
from atria_insights.feature_segmentors._image import (
    FelzenszwalbImageSegmenterConfig,
    GridSegmenter,
    GridSegmenterConfig,
    ImageSegmentorConfigType,
    QuickshiftImageSegmenterConfig,
    ScikitImageSegmenter,
    SlicImageSegmenterConfig,
)
from atria_insights.feature_segmentors._sequence import (
    SequenceFeatureMaskSegmentor,
    SequenceFeatureMaskSegmentorConfig,
)

FeatureSegmentorConfigType = Annotated[
    ImageSegmentorConfigType, Field(discriminator="type")
]
__all__ = [
    "NoOpSegmenterConfig",
    "FeatureSegmentorConfigType",
    "ImageSegmentorConfigType",
    "GridSegmenterConfig",
    "SlicImageSegmenterConfig",
    "QuickshiftImageSegmenterConfig",
    "FelzenszwalbImageSegmenterConfig",
    "ScikitImageSegmenter",
    "GridSegmenter",
    "SequenceFeatureMaskSegmentorConfig",
    "SequenceFeatureMaskSegmentor",
]
