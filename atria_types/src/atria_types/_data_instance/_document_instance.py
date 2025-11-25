from typing import Self

from pydantic import model_validator

from atria_types._data_instance._base import BaseDataInstance
from atria_types._generic._image import Image
from atria_types._generic._ocr import OCR
from atria_types._generic._pdf import PDF
from atria_types._generic._text_elements import TextElement
from atria_types.ocr_parsers.hocr_parser import OCRProcessor


class DocumentInstance(BaseDataInstance):
    page_id: int | None = None
    pdf: PDF | None = None
    image: Image | None = None
    ocr: OCR | None = None
    text_elements: list[TextElement] | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> "DocumentInstance":
        # Ensure we have either image or PDF
        if self.image is None and self.pdf is None:
            raise ValueError("Either image or pdf must be provided")

    def load(
        self,
    ) -> Self:  # this is the top level load that loads all children fields _load
        loaded_fields = {
            name: self._load(getattr(self, name))
            for name in self.__class__.model_fields
        }

        if loaded_fields["ocr"] is not None and loaded_fields["text_elements"] is None:
            loaded_fields["text_elements"] = OCRProcessor.parse(
                raw_ocr=self.ocr.content, ocr_type=self.ocr.type
            )

        new_instance = self.model_copy(update=loaded_fields)
        return new_instance
