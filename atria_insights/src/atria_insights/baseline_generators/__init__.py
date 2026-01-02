from __future__ import annotations

from typing import Annotated

from pydantic import Field

from atria_insights.baseline_generators._feature_based import (
    FeatureBasedBaselineGeneratorConfig,
)
from atria_insights.baseline_generators._sequence import SequenceBaselineGeneratorConfig

from ._simple import SimpleBaselineGeneratorConfig

BaselineGeneratorConfigType = Annotated[
    SimpleBaselineGeneratorConfig | FeatureBasedBaselineGeneratorConfig,
    Field(discriminator="type"),
]

__all__ = [
    "BaselineGeneratorConfigType",
    "SimpleBaselineGeneratorConfig",
    "SequenceBaselineGeneratorConfig",
    "FeatureBasedBaselineGeneratorConfig",
]
