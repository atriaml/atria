from enum import Enum


class DatasetStatus(str, Enum):
    AVAILABLE = "available"
    OUTDATED = "outdated"
    UNAVAILABLE = "unavailable"

    def __str__(self) -> str:
        return str(self.value)
