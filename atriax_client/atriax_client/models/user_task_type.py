from enum import Enum


class UserTaskType(str, Enum):
    DATASET_VALIDATION = "dataset_validation"
    EVALUATION = "evaluation"
    EXPLANATION = "explanation"
    MODEL_VALIDATION = "model_validation"

    def __str__(self) -> str:
        return str(self.value)
