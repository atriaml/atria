import torch
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa: F401
from atria_logger import get_logger
from pydantic import BaseModel, ConfigDict

logger = get_logger(__name__)

Primitives = float | int | str
AttributeType = Primitives | list[Primitives] | None


class SerializableSampleData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    sample_id: str
    attrs: dict[str, AttributeType] | None = None
    tensors: dict[str, dict[str, torch.Tensor] | torch.Tensor | list[str]] | None = None
