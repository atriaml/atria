from __future__ import annotations

from atria_logger import get_logger

from atria_types._generic._bounding_box import BoundingBox, BoundingBoxMode

logger = get_logger(__name__)


class BoundingBoxOps:
    """Operations for BoundingBox

    A mixin class that provides operations for BoundingBox.
    """

    def __init__(self, bbox: BoundingBox):
        self.bbox = bbox

    def switch_mode(self) -> BoundingBox:
        """
        Switches the bounding box mode between XYXY and XYWH.
        Returns:
            BoundingBox: A new BoundingBox instance with the switched mode.
        """
        if self.bbox.mode == BoundingBoxMode.XYXY:
            new_value = [self.bbox.x1, self.bbox.y1, self.bbox.width, self.bbox.height]
            new_mode = BoundingBoxMode.XYWH
        else:
            new_value = [self.bbox.x1, self.bbox.y1, self.bbox.x2, self.bbox.y2]
            new_mode = BoundingBoxMode.XYXY
        return self.bbox.model_copy(
            update={
                "value": new_value,
                "mode": new_mode,
            }
        )

    def normalize(self, width: float, height: float) -> BoundingBox:
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
        assert self.bbox.x1 <= width, "x1 must be less than or equal to width."
        assert self.bbox.y1 <= height, "y1 must be less than or equal to height."
        assert self.bbox.x2 <= width, "x2 must be less than or equal to width."
        assert self.bbox.y2 <= height, "y2 must be less than or equal to height."
        if self.bbox.mode == BoundingBoxMode.XYWH:
            return BoundingBox(
                value=[
                    self.bbox.x1 / width,
                    self.bbox.y1 / height,
                    self.bbox.width / width,
                    self.bbox.height / height,
                ],
                mode=self.bbox.mode,
                normalized=True,
            )
        else:
            return BoundingBox(
                value=[
                    self.bbox.x1 / width,
                    self.bbox.y1 / height,
                    self.bbox.x2 / width,
                    self.bbox.y2 / height,
                ],
                mode=self.bbox.mode,
                normalized=True,
            )
