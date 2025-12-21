from typing import Annotated

from pydantic import Field

from atria_types._ocr_engines._tesseract import TesseractOCREngineConfig

OCREngineConfigType = Annotated[
    TesseractOCREngineConfig,
    Field(discriminator="type"),
]
