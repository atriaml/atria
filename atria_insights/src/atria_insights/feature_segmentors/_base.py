from __future__ import annotations

from abc import abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Generic, Literal

import torch
from atria_registry._module_base import ConfigurableModule, T_ModuleConfig



class FeatureSegmentor(ConfigurableModule[T_ModuleConfig], Generic[T_ModuleConfig]):
    __abstract__ = True

    @abstractmethod
    def __call__(
        self, inputs: torch.Tensor | OrderedDict[str, torch.Tensor], **kwargs: Any
    ) -> torch.Tensor | OrderedDict[str, torch.Tensor]: ...
