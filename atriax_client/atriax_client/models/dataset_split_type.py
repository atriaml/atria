from enum import Enum


class DatasetSplitType(str, Enum):
    TEST = "test"
    TRAIN = "train"
    VALIDATION = "validation"

    def __str__(self) -> str:
        return str(self.value)
