from __future__ import annotations

from abc import abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Generic

from atria_registry._module_base import ConfigurableModule, T_ModuleConfig

if TYPE_CHECKING:
    import torch
    import torch.nn as nn


class BaselineGenerator(ConfigurableModule[T_ModuleConfig], Generic[T_ModuleConfig]):
    __abstract__ = True

    def __init__(
        self,
        model: nn.Module | None = None,
        config: T_ModuleConfig | dict | None = None,
    ) -> None:
        super().__init__(config)
        self._model = model

    @abstractmethod
    def __call__(
        self, inputs: torch.Tensor | OrderedDict[str, torch.Tensor], **kwargs: Any
    ) -> torch.Tensor | OrderedDict[str, torch.Tensor]: ...
