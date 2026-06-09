from __future__ import annotations

from typing import Annotated, Literal, Union

import numpy as np
import torch
from atria_logger import get_logger
from atria_types._utilities._repr import RepresentationMixin
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from atria_insights.utilities._viz import score_to_color_map

logger = get_logger(__name__)

_UNIT_CONFIG = ConfigDict(
    arbitrary_types_allowed=True,
    validate_assignment=True,
    frozen=True,
    extra="forbid",
)

_CONTAINER_CONFIG = ConfigDict(
    arbitrary_types_allowed=True,
    validate_assignment=True,
    frozen=True,
    extra="forbid",
    revalidate_instances="always",
)


def _to_color_array(v: torch.Tensor | np.ndarray | list | None) -> np.ndarray | None:
    if v is None:
        return None
    if isinstance(v, torch.Tensor):
        return score_to_color_map(v.detach().cpu().numpy())
    if isinstance(v, list):
        return np.array(v)
    return v  # already np.ndarray


class TextExplanationUnit(BaseModel):
    model_config = _UNIT_CONFIG
    unit_type: Literal["text"] = "text"
    attribution: np.ndarray
    context_attribution: np.ndarray | None = None

    @field_validator("attribution", mode="before")
    @classmethod
    def _val_attribution(cls, v: torch.Tensor | np.ndarray | list) -> np.ndarray:
        if isinstance(v, torch.Tensor):
            return score_to_color_map(v.detach().cpu().numpy())
        if isinstance(v, list):
            return np.array(v)
        return v

    @field_validator("context_attribution", mode="before")
    @classmethod
    def _val_context(cls, v: torch.Tensor | np.ndarray | list | None) -> np.ndarray | None:
        return _to_color_array(v)

    @field_serializer("attribution", "context_attribution")
    def _ser_array(self, v: np.ndarray | None) -> list | None:
        return v.tolist() if v is not None else None

    @property
    def name(self) -> str:
        return "Text"


class TextPositionExplanationUnit(TextExplanationUnit):
    unit_type: Literal["text_position"] = "text_position"

    @property
    def name(self) -> str:
        return "Position"


class TextLayoutExplanationUnit(TextExplanationUnit):
    unit_type: Literal["text_layout"] = "text_layout"

    @property
    def name(self) -> str:
        return "Layout"


class AggregateTextExplanationUnit(TextExplanationUnit):
    unit_type: Literal["text_aggregate"] = "text_aggregate"

    @property
    def name(self) -> str:
        return "Agg. Text"


class ImageExplanationUnit(BaseModel):
    model_config = _UNIT_CONFIG
    unit_type: Literal["image"] = "image"
    attribution: np.ndarray

    @field_validator("attribution", mode="before")
    @classmethod
    def _val_attribution(cls, v: torch.Tensor | np.ndarray | list) -> np.ndarray:
        if isinstance(v, torch.Tensor):
            return score_to_color_map(v.detach().cpu().numpy())
        if isinstance(v, list):
            return np.array(v)
        return v

    @field_serializer("attribution")
    def _ser_array(self, v: np.ndarray) -> list:
        return v.tolist()

    @property
    def name(self) -> str:
        return "Image"


ExplanationUnit = Annotated[
    Union[
        TextExplanationUnit,
        TextPositionExplanationUnit,
        TextLayoutExplanationUnit,
        AggregateTextExplanationUnit,
        ImageExplanationUnit,
    ],
    Field(discriminator="unit_type"),
]


class SampleExplanationSummary(RepresentationMixin, BaseModel):
    """All ExplanationUnit objects for one sample and one target (one per feature)."""

    model_config = _CONTAINER_CONFIG
    value: list[ExplanationUnit]

    @property
    def n_units(self) -> int:
        return len(self.value)


class BatchExplanationSummary(RepresentationMixin, BaseModel):
    """Single-target explanation summaries for a full batch (one SampleExplanationSummary per sample)."""

    model_config = _CONTAINER_CONFIG
    value: list[SampleExplanationSummary]

    @property
    def batch_size(self) -> int:
        return len(self.value)

    def tolist(self) -> list[SampleExplanationSummary]:
        return list(self.value)


class MultiTargetSampleExplanationSummary(RepresentationMixin, BaseModel):
    """Multi-target explanation summaries for one sample (one SampleExplanationSummary per target)."""

    model_config = _CONTAINER_CONFIG
    value: list[SampleExplanationSummary]

    @property
    def n_targets(self) -> int:
        return len(self.value)


class MultiTargetBatchExplanationSummary(RepresentationMixin, BaseModel):
    """Multi-target explanation summaries for a full batch (one BatchExplanationSummary per target)."""

    model_config = _CONTAINER_CONFIG
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
