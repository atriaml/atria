from enum import Enum


class NormalizationType(str, Enum):
    ABSOLUTE_VALUE = "absolute_value"
    ALL = "all"
    NEGATIVE = "negative"
    POSITIVE = "positive"

    def __str__(self) -> str:
        return str(self.value)
