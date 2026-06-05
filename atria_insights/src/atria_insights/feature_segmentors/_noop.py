from __future__ import annotations

from abc import abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Generic, Literal

from atria_insights.feature_segmentors._base import FeatureSegmentor
from atria_insights.feature_segmentors._image import ScikitImageSegmenter, SlicImageSegmenterConfig
import torch
from atria_registry._module_base import ConfigurableModule, ModuleConfig, T_ModuleConfig



class NoOpSegmenterConfig(ModuleConfig):
    type: Literal["noop"] = "noop"

    def build(self, **kwargs: Any) -> Callable:
        return lambda x: None

class NoOpSegmenter(FeatureSegmentor[NoOpSegmenterConfig]):
    __config__ = NoOpSegmenterConfig

    def __call__(
        self, inputs: torch.Tensor | OrderedDict[str, torch.Tensor], **kwargs: Any
    ) -> torch.Tensor | OrderedDict[str, torch.Tensor]:
        return inputs