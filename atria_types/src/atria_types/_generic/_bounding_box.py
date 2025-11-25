import enum
from typing import Annotated, Any

from atria_logger import get_logger
from pydantic import field_validator

from atria_types._base._data_model import BaseDataModel
from atria_types._pydantic import (
    ListFloatField,
    TableSchemaMetadata,
    _is_tensor_type,
)

logger = get_logger(__name__)


class BoundingBoxMode(str, enum.Enum):
    XYXY = "xyxy"  # (x1, y1, x2, y2)
    XYWH = "xywh"  # (x1, y1, width, height)


class BoundingBox(BaseDataModel):
    value: ListFloatField
    mode: Annotated[BoundingBoxMode, TableSchemaMetadata(pa_type="string")] = (
        BoundingBoxMode.XYXY
    )
    normalized: bool = False

    def switch_mode(self) -> "BoundingBox":
        if self.mode == BoundingBoxMode.XYXY:
            new_value = [self.x1, self.y1, self.width, self.height]
            return BoundingBox(
                value=new_value,
                mode=BoundingBoxMode.XYWH,
                normalized=self.normalized,
            )
        else:
            new_value = [self.x1, self.y1, self.x2, self.y2]
            return BoundingBox(
                value=new_value,
                mode=BoundingBoxMode.XYXY,
                normalized=self.normalized,
            )

    @field_validator("mode", mode="before")
    @classmethod
    def validate_mode(cls, value: Any) -> BoundingBoxMode:
        if isinstance(value, str):
            return BoundingBoxMode(value)
        return value

    @field_validator("value", mode="after")
    @classmethod
    def validate_value(cls, value: Any) -> list[float]:
        assert len(value) == 4, "Expected a 1D list of shape (4,) for bounding boxes."
        return value

    @property
    def is_valid(self) -> bool:
        return (
            self.x1 >= 0
            and self.y1 >= 0
            and self.x2 > self.x1
            and self.y2 > self.y1
            and self.width > 0
            and self.height > 0
        )

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def x1(self) -> float:
        idx = (..., 0) if _is_tensor_type(self.value) else 0
        return self.value[idx]

    @x1.setter
    def x1(self, value: float):
        idx = (..., 0) if _is_tensor_type(self.value) else 0
        self.value[idx] = value

    @property
    def y1(self) -> float:
        idx = (..., 1) if _is_tensor_type(self.value) else 1
        return self.value[idx]

    @y1.setter
    def y1(self, value: float):
        idx = (..., 1) if _is_tensor_type(self.value) else 1
        self.value[idx] = value

    @property
    def x2(self) -> float:
        if self.mode == BoundingBoxMode.XYWH:
            return self.x1 + self.width
        else:
            idx = (..., 2) if _is_tensor_type(self.value) else 2
            return self.value[idx]

    @x2.setter
    def x2(self, value: float):
        if self.mode == BoundingBoxMode.XYWH:
            raise ValueError("Cannot set x2 directly in XYWH mode. Use width instead.")
        else:
            idx = (..., 2) if _is_tensor_type(self.value) else 2
            self.value[idx] = value

    @property
    def y2(self) -> float:
        if self.mode == BoundingBoxMode.XYWH:
            return self.y1 + self.height
        else:
            idx = (..., 3) if _is_tensor_type(self.value) else 3
            return self.value[idx]

    @y2.setter
    def y2(self, value: float):
        if self.mode == BoundingBoxMode.XYWH:
            raise ValueError("Cannot set x2 directly in XYWH mode. Use width instead.")
        else:
            idx = (..., 3) if _is_tensor_type(self.value) else 3
            self.value[idx] = value

    @property
    def width(self) -> float:
        if self.mode == BoundingBoxMode.XYWH:
            idx = (..., 2) if _is_tensor_type(self.value) else 2
            return self.value[idx]
        else:
            return self.x2 - self.x1

    @width.setter
    def width(self, value: float):
        if self.mode == BoundingBoxMode.XYWH:
            idx = (..., 2) if _is_tensor_type(self.value) else 2
            self.value[idx] = value
        else:
            raise ValueError("Cannot set width directly in XYXY mode. Use x2 instead.")

    @property
    def height(self) -> float:
        if self.mode == BoundingBoxMode.XYWH:
            idx = (..., 3) if _is_tensor_type(self.value) else 3
            return self.value[idx]
        else:
            return self.y2 - self.y1

    @height.setter
    def height(self, value: float):
        if self.mode == BoundingBoxMode.XYWH:
            idx = (..., 3) if _is_tensor_type(self.value) else 3
            self.value[idx] = value
        else:
            raise ValueError("Cannot set height directly in XYXY mode. Use y2 instead.")

    def normalize(self, width: float, height: float) -> "BoundingBox":
        """
        Normalizes the bounding box coordinates to the range [0, 1] given the image width and height.

        Args:
            width (float): The width of the image or document.
            height (float): The height of the image or document.

        Returns:
            BoundingBox: The normalized bounding box.

        Raises:
            AssertionError: If the bounding box coordinates are invalid.
        """
        assert width > 0, "Width must be greater than 0."
        assert height > 0, "Height must be greater than 0."
        assert self.x1 <= width, "x1 must be less than or equal to width."
        assert self.y1 <= height, "y1 must be less than or equal to height."
        assert self.x2 <= width, "x2 must be less than or equal to width."
        assert self.y2 <= height, "y2 must be less than or equal to height."
        if self.mode == BoundingBoxMode.XYWH:
            return BoundingBox(
                value=[
                    self.x1 / width,
                    self.y1 / height,
                    self.width / width,
                    self.height / height,
                ],
                mode=self.mode,
                normalized=True,
            )
        else:
            return BoundingBox(
                value=[
                    self.x1 / width,
                    self.y1 / height,
                    self.x2 / width,
                    self.y2 / height,
                ],
                mode=self.mode,
                normalized=True,
            )
