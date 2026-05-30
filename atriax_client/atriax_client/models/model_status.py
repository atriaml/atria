from enum import Enum


class ModelStatus(str, Enum):
    CREATED = "created"
    UPLOADED = "uploaded"
    VALIDATED = "validated"
    VALIDATION_FAILED = "validation_failed"

    def __str__(self) -> str:
        return str(self.value)
