from __future__ import annotations

from typing import Annotated

from pydantic import Field

from atria_insights.model_pipelines.baseline_generators._sequence import (
    SequenceBaselineGeneratorConfig,
)

from ._simple import SimpleBaselineGeneratorConfig

BaselineGeneratorConfigType = Annotated[
    SimpleBaselineGeneratorConfig | SequenceBaselineGeneratorConfig,
    Field(discriminator="type"),
]
