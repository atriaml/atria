from enum import Enum


class DataInstanceType(str, Enum):
    DOCUMENT_INSTANCE = "document_instance"
    IMAGE_INSTANCE = "image_instance"

    def __str__(self) -> str:
        return str(self.value)
