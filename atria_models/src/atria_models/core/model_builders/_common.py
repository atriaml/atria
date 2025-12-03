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
    transformers_sequence = "transformers/sequence_classification"
    transformers_token_classification = "transformers/token_classification"
    transformers_question_answering = "transformers/question_answering"
    transformers_image_classification = "transformers/image_classification"
