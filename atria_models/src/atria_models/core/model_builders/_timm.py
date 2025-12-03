"""Timm model builders."""

from typing import TYPE_CHECKING

from atria_logger import get_logger
from rich.pretty import pretty_repr

from atria_models.core.model_builders._base import ModelBuilder

if TYPE_CHECKING:
    from torch import nn

logger = get_logger(__name__)


class TimmModelBuilder(ModelBuilder):
    def _build(self, model_name_or_path: str, **kwargs) -> nn.Module:
        import timm

        filtered_kwargs = {"model_name": model_name_or_path, **kwargs}
        if "num_labels" in filtered_kwargs:
            num_labels = kwargs["num_labels"]
            filtered_kwargs["num_classes"] = num_labels
        logger.info(
            f"Building model '{model_name_or_path}' with parameters:\n{pretty_repr(filtered_kwargs, expand_all=True)}"
        )
        return timm.create_model(**filtered_kwargs)
