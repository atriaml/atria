from enum import Enum


class UserTaskType(str, Enum):
    DATASETE_PREPROCESSING = "datasete_preprocessing"
    DATASET_VALIDATION = "dataset_validation"
    EVALUATION = "evaluation"
    EXPLANATION = "explanation"
    INFERENCE = "inference"
    MODEL_VALIDATION = "model_validation"

    def __str__(self) -> str:
        return str(self.value)
