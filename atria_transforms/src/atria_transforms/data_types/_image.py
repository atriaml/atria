from __future__ import annotations

import torch
from pydantic import BaseModel

from atria_transforms.core import TensorDataModel


class ImageTensorDataModel(TensorDataModel):
    class Metadata(BaseModel):
        index: int | None
        sample_id: str

    # sample level fields
    image: torch.Tensor
    label: torch.Tensor | None = None
