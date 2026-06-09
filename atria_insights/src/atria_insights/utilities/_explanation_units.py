from __future__ import annotations

from typing import ClassVar

import torch
from atria_logger import get_logger
from atria_types._utilities._repr import RepresentationMixin
from pydantic import BaseModel, ConfigDict, model_serializer

from atria_insights.utilities._viz import score_to_color_map

logger = get_logger(__name__)

_MODEL_CONFIG = ConfigDict(
    arbitrary_types_allowed=True,
    validate_assignment=True,
    frozen=True,
    extra="forbid",
    revalidate_instances="always",
)


class TextExplanationUnit:
    unit_type: ClassVar[str] = "text"

    def __init__(
        self, attribution: torch.Tensor, context_attribution: torch.Tensor | None = None
    ):
        self.attribution = score_to_color_map(attribution.detach().cpu().numpy())
        self.context_attribution = (
            score_to_color_map(context_attribution.detach().cpu().numpy())
            if context_attribution is not None
            else None
        )

    @property
    def name(self):
        return "Text"


class TextPositionExplanationUnit(TextExplanationUnit):
    unit_type: ClassVar[str] = "text_position"

    @property
    def name(self) -> str:
        return "Position"


class TextLayoutExplanationUnit(TextExplanationUnit):
    unit_type: ClassVar[str] = "text_layout"

    @property
    def name(self) -> str:
        return "Layout"


class AggregateTextExplanationUnit(TextExplanationUnit):
    unit_type: ClassVar[str] = "text_aggregate"

    @property
    def name(self) -> str:
        return "Agg. Text"


class ImageExplanationUnit:
    unit_type: ClassVar[str] = "image"

    def __init__(self, attribution: torch.Tensor):
        self.attribution = score_to_color_map(attribution.detach().cpu().numpy())

    @property
    def name(self) -> str:
        return "Image"


ExplanationUnit = (
    TextExplanationUnit
    | TextPositionExplanationUnit
    | TextLayoutExplanationUnit
    | AggregateTextExplanationUnit
    | ImageExplanationUnit
)


class SampleExplanationSummary(RepresentationMixin, BaseModel):
    """All ExplanationUnit objects for one sample and one target (one per feature)."""

    model_config = _MODEL_CONFIG
    value: list[ExplanationUnit]

    @property
    def n_units(self) -> int:
        return len(self.value)

    @model_serializer
    def _serialize(self) -> dict:
        units = []
        for unit in self.value:
            entry = {"unit_type": unit.unit_type}
            for attr, val in vars(unit).items():
                entry[attr] = val.tolist() if hasattr(val, "tolist") else val
            units.append(entry)
        return {"value": units}


class BatchExplanationSummary(RepresentationMixin, BaseModel):
    """Single-target explanation summaries for a full batch (one SampleExplanationSummary per sample)."""

    model_config = _MODEL_CONFIG
    value: list[SampleExplanationSummary]

    @property
    def batch_size(self) -> int:
        return len(self.value)

    def tolist(self) -> list[SampleExplanationSummary]:
        return list(self.value)


class MultiTargetSampleExplanationSummary(RepresentationMixin, BaseModel):
    """Multi-target explanation summaries for one sample (one SampleExplanationSummary per target)."""

    model_config = _MODEL_CONFIG
    value: list[SampleExplanationSummary]

    @property
    def n_targets(self) -> int:
        return len(self.value)


class MultiTargetBatchExplanationSummary(RepresentationMixin, BaseModel):
    """Multi-target explanation summaries for a full batch (one BatchExplanationSummary per target)."""

    model_config = _MODEL_CONFIG
    value: list[BatchExplanationSummary]

    @property
    def n_targets(self) -> int:
        return len(self.value)

    @property
    def batch_size(self) -> int:
        return self.value[0].batch_size if self.value else 0

    def tolist(self) -> list[MultiTargetSampleExplanationSummary]:
        per_target_lists = [batch.tolist() for batch in self.value]
        return [
            MultiTargetSampleExplanationSummary(
                value=[per_target_lists[t][i] for t in range(self.n_targets)]
            )
            for i in range(self.batch_size)
        ]
