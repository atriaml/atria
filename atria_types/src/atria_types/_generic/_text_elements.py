from atria_types._base._data_model import BaseDataModel
from atria_types._generic._bounding_box import BoundingBox
from atria_types._pydantic import OptFloatField, OptStrField


class TextElement(BaseDataModel):
    value: OptStrField = None
    bbox: BoundingBox | None = None
    segment_bbox: BoundingBox | None = None
    conf: OptFloatField = None
    angle: OptFloatField = None
