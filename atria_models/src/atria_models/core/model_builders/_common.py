"""Common Dataset Types and Enums"""

from __future__ import annotations

import enum

from atria_logger import get_logger

logger = get_logger(__name__)


class FrozenLayers(str, enum.Enum):
    none = "none"
    all = "all"


class ModelBuilderType(str, enum.Enum):
    local = "local"
    timm = "timm"
    torchvision = "torchvision"
    transformers = "transformers"
    atria = "atria"
