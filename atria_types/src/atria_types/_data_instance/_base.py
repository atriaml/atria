import uuid
from typing import Annotated, Literal, overload

from pydantic import Field, field_serializer, field_validator

from atria_types._base._data_model import BaseDataModel
from atria_types._data_instance._exceptions import AnnotationNotFoundError
from atria_types._generic._annotations import (
    Annotation,
    AnnotationType,
    ClassificationAnnotation,
    EntityLabelingAnnotation,
    LayoutAnalysisAnnotation,
    ObjectDetectionAnnotation,
    QuestionAnsweringAnnotation,
)
from atria_types._pydantic import OptIntField, StrField, TableSchemaMetadata


class BaseDataInstance(BaseDataModel):
    index: OptIntField = None
    sample_id: StrField = Field(default_factory=lambda: str(uuid.uuid4()))
    annotations: Annotated[
        list[Annotation] | None, TableSchemaMetadata(pa_type="string")
    ] = None

    @property
    def key(self) -> str:
        """
        Generates a unique key for the data instance.

        The key is a combination of the UUID and the index (if present).

        Returns:
            str: The unique key for the data instance.
        """
        return str(self.sample_id.replace(".", "_"))

    @overload
    def get_annotation_by_type(
        self, annotation_type: Literal[AnnotationType.classification]
    ) -> ClassificationAnnotation: ...

    @overload
    def get_annotation_by_type(
        self, annotation_type: Literal[AnnotationType.entity_labeling]
    ) -> EntityLabelingAnnotation: ...

    @overload
    def get_annotation_by_type(
        self, annotation_type: Literal[AnnotationType.question_answering]
    ) -> QuestionAnsweringAnnotation: ...

    @overload
    def get_annotation_by_type(
        self, annotation_type: Literal[AnnotationType.object_detection]
    ) -> ObjectDetectionAnnotation: ...

    @overload
    def get_annotation_by_type(
        self, annotation_type: Literal[AnnotationType.layout_analysis]
    ) -> LayoutAnalysisAnnotation: ...

    def get_annotation_by_type(self, annotation_type: AnnotationType) -> Annotation:
        """
        Retrieves the first annotation of the specified type.

        Args:
            annotation_type (AnnotationType): The type of annotation to retrieve.

        Returns:
            Annotation: The first annotation of the specified type.
        """
        if self.annotations is not None:
            for annotation in self.annotations:
                if annotation._type == annotation_type:
                    return annotation
        raise AnnotationNotFoundError(
            f"No annotation of type {annotation_type} found in the data instance."
        )

    @field_validator("annotations", mode="before")
    def validate_annotations(cls, value) -> list[Annotation] | None:
        if isinstance(value, str):
            import json

            try:
                value = json.loads(value)
                assert isinstance(value, list), "Annotations JSON must be a list"

                annotations = []
                for item in value:
                    assert isinstance(item, dict), "Each annotation must be a dict"

                    ann_type = item.pop("type", None)
                    annotations.append(Annotation.from_type(ann_type, item))
                return annotations
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON string: {value}") from None

        return value

    @field_serializer("annotations")
    def serialize_annotations(self, value: list[Annotation]) -> str | None:
        import json

        if value is not None:
            return json.dumps(
                [
                    item.model_dump() if hasattr(item, "model_dump") else item
                    for item in value
                ]
            )
        return None
