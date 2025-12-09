"""Timm model builders."""

from __future__ import annotations

from typing import TYPE_CHECKING

from atria_logger import get_logger

from atria_models.core.model_builders._base import ModelBuilder, pretty_kwargs

if TYPE_CHECKING:
    from torch import nn

logger = get_logger(__name__)


class TimmModelBuilder(ModelBuilder):
    def _build(
        self, model_name_or_path: str, pretrained: bool = True, **kwargs
    ) -> nn.Module:
        import timm

        filtered_kwargs = {
            "model_name": model_name_or_path,
            "pretrained": pretrained,
            **kwargs,
        }
        if "num_labels" in filtered_kwargs:
            filtered_kwargs["num_classes"] = filtered_kwargs.pop("num_labels")
        logger.info(
            f"Building model '{model_name_or_path}' with parameters:{pretty_kwargs(filtered_kwargs)}"
        )
        return timm.create_model(**filtered_kwargs)
