"""Torchvision model builders."""

import os
import typing
from typing import TYPE_CHECKING

from atria_logger import get_logger
from rich.pretty import pretty_repr

from atria_models.core.model_builders._base import ModelBuilder
from atria_models.utilities._nn_modules import (
    _get_last_module,
    _replace_module_with_name,
)

if TYPE_CHECKING:
    from torch import nn

logger = get_logger(__name__)


class TorchvisionModelBuilder(ModelBuilder):
    def _build(self, model_name_or_path: str, **kwargs) -> nn.Module:
        import torch

        os.environ["TORCH_HOME"] = str(self._cache_dir)

        logger.info(
            f"Building model '{model_name_or_path}' with parameters:\n{pretty_repr(kwargs, expand_all=True)}"
        )
        model = typing.cast(
            nn.Module,
            torch.hub.load(
                "pytorch/vision:v0.10.0", model_name_or_path, verbose=False, **kwargs
            ),
        )

        if "num_labels" in kwargs:
            num_labels = kwargs["num_labels"]
            from torch.nn import Linear

            name, module = _get_last_module(model)
            _replace_module_with_name(
                model, name, Linear(module.in_features, num_labels)
            )
        else:
            logger.warning(
                "No 'num_labels' in 'model_initialization_kwargs' provided. "
                "Classification head will not be replaced."
            )

        return model
