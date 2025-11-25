from typing import Annotated

from pydantic import field_serializer, field_validator

from atria_types._base._data_model import BaseDataModel
from atria_types._generic._bounding_box import BoundingBox
from atria_types._pydantic import OptFloatField, OptStrField, TableSchemaMetadata


class TextElement(BaseDataModel):
    text: OptStrField = None
    bbox: BoundingBox | None = None
    segment_bbox: BoundingBox | None = None
    conf: OptFloatField = None
    angle: OptFloatField = None


class DocumentContent(BaseDataModel):
    text_elements: Annotated[
        list[TextElement], TableSchemaMetadata(pa_type="string")
    ] = None

    @field_validator("text_elements", mode="before")
    def validate_text_elements(cls, value) -> list[TextElement]:
        if isinstance(value, str):
            import json

            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON string: {value}") from None
        return value

    @field_serializer("text_elements")
    def serialize_text_elements(self, value: list[TextElement]) -> str:
        if value is None:
            return None

        import json

        return json.dumps([item.model_dump() for item in value])
