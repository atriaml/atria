from enum import Enum


class TaskType(str, Enum):
    IMAGE_CLASSIFICATION = "image_classification"
    LAYOUT_ANALYSIS = "layout_analysis"
    LAYOUT_TOKEN_CLASSIFICATION = "layout_token_classification"
    OBJECT_DETECTION = "object_detection"
    QUESTION_ANSWERING = "question_answering"
    SEQUENCE_CLASSIFICATION = "sequence_classification"
    TOKEN_CLASSIFICATION = "token_classification"
    VISUAL_QUESTION_ANSWERING = "visual_question_answering"

    def __str__(self) -> str:
        return str(self.value)
