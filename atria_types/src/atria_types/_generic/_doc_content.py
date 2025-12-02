from typing import Annotated, Any

from pydantic import field_serializer, field_validator, model_validator

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
    text: OptStrField = None
    text_elements: Annotated[
        list[TextElement] | None, TableSchemaMetadata(pa_type="string")
    ] = None

    @property
    def text_list(self) -> list[str]:
        if self.text_elements is None:
            return []
        return [te.text for te in self.text_elements if te.text is not None]

    @property
    def bbox_list(self) -> list[BoundingBox]:
        if self.text_elements is None:
            return []
        return [te.bbox for te in self.text_elements if te.bbox is not None]

    @property
    def segment_bbox_list(self) -> list[BoundingBox]:
        if self.text_elements is None:
            return []
        return [
            te.segment_bbox for te in self.text_elements if te.segment_bbox is not None
        ]

    @model_validator(mode="before")
    def validate_content(cls, values: Any) -> Any:
        if values.get("text") is None and values.get("text_elements") is not None:
            texts = [te.text for te in values["text_elements"] if te.text is not None]
            values["text"] = "\n".join(texts)
        return values

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
