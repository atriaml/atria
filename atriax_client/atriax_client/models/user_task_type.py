from enum import Enum


class UserTaskType(str, Enum):
    EVALUATION = "evaluation"
    EXPLANATION = "explanation"

    def __str__(self) -> str:
        return str(self.value)
