from __future__ import annotations

import enum
from typing import Annotated

from pydantic import field_serializer, field_validator

from atria_types._base._data_model import BaseDataModel
from atria_types._generic._annotated_object import AnnotatedObject
from atria_types._generic._label import Label
from atria_types._generic._qa_pair import QAPair
from atria_types._pydantic import TableSchemaMetadata


class AnnotationType(str, enum.Enum):
    classification = "classification"
    entity_labeling = "entity_labeling"
    question_answering = "question_answering"
    object_detection = "object_detection"
    layout_analysis = "layout_analysis"


class Annotation(BaseDataModel):
    _type: AnnotationType

    @classmethod
    def from_type(cls, annotation_type: AnnotationType, params: dict) -> Annotation:
        if annotation_type == AnnotationType.classification:
            return ClassificationAnnotation(**params)
        elif annotation_type == AnnotationType.entity_labeling:
            return EntityLabelingAnnotation(**params)
        elif annotation_type == AnnotationType.question_answering:
            return QuestionAnsweringAnnotation(**params)
        elif annotation_type == AnnotationType.object_detection:
            return ObjectDetectionAnnotation(**params)
        elif annotation_type == AnnotationType.layout_analysis:
            return LayoutAnalysisAnnotation(**params)
        else:
            raise ValueError(f"Unknown annotation type: {annotation_type}")

    def model_dump(self, *args, **kwargs):
        return {**super().model_dump(*args, **kwargs), "type": self._type}


class ClassificationAnnotation(Annotation):
    _type: AnnotationType = AnnotationType.classification
    label: Label


class EntityLabelingAnnotation(Annotation):
    _type: AnnotationType = AnnotationType.entity_labeling
    word_labels: list[Label]

    @field_validator("word_labels", mode="before")
    def validate_word_labels(cls, value) -> list[Label] | None:
        if isinstance(value, str):
            import json

            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON string: {value}") from None
        return value

    @field_serializer("word_labels")
    def serialize_word_labels(self, value: list[Label]) -> str | None:
        import json

        if value is not None:
            return json.dumps([item.model_dump() for item in value])
        return None


class QuestionAnsweringAnnotation(Annotation):
    _type: AnnotationType = AnnotationType.question_answering
    qa_pairs: Annotated[list[QAPair] | None, TableSchemaMetadata(pa_type="string")] = (
        None
    )

    @field_validator("qa_pairs", mode="before")
    def validate_qa_pairs(cls, value) -> list[QAPair] | None:
        if isinstance(value, str):
            import json

            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON string: {value}") from None
        return value

    @field_serializer("qa_pairs")
    def serialize_qa_pairs(self, value: list[QAPair]) -> str | None:
        import json

        if value is not None:
            return json.dumps([item.model_dump() for item in value])
        return None


class ObjectDetectionAnnotation(Annotation):
    _type: AnnotationType = AnnotationType.object_detection
    annotated_objects: Annotated[
        list[AnnotatedObject] | None, TableSchemaMetadata(pa_type="string")
    ] = None

    @field_validator("annotated_objects", mode="before")
    def validate_annotated_objects(cls, value) -> list[AnnotatedObject] | None:
        if isinstance(value, str):
            import json

            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON string: {value}") from None
        return value

    @field_serializer("annotated_objects")
    def serialize_annotated_objects(self, value: list[AnnotatedObject]) -> str | None:
        import json

        if value is not None:
            return json.dumps([item.model_dump() for item in value])
        return None


class LayoutAnalysisAnnotation(ObjectDetectionAnnotation):
    _type: AnnotationType = AnnotationType.layout_analysis
