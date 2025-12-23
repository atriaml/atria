from typing import Any

import torch
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa: F401
from atria_logger import get_logger
from pydantic import BaseModel, ConfigDict

logger = get_logger(__name__)


class CacheData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    sample_id: str
    attrs: dict[str, Any] | None = None
    tensors: dict[str, dict[str, torch.Tensor] | torch.Tensor] | None = None
