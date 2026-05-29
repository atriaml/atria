from enum import Enum


class ExplanationOutputType(str, Enum):
    HEATMAP = "heatmap"

    def __str__(self) -> str:
        return str(self.value)
