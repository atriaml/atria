from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from atria_transforms.core import TensorDataModel


class ImageTensorDataModel(TensorDataModel):
    class Metadata(BaseModel):
        index: int | None
        sample_id: str

    # sample level fields
    image: np.ndarray
    label: np.ndarray | None = None
