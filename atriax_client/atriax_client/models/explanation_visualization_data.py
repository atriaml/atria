from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.explanation_visualization_data_outputs_type_0 import ExplanationVisualizationDataOutputsType0
    from ..models.explanation_visualization_data_processing_options_type_0 import (
        ExplanationVisualizationDataProcessingOptionsType0,
    )


T = TypeVar("T", bound="ExplanationVisualizationData")


@_attrs_define
class ExplanationVisualizationData:
    """Base output structure for explanation visualizers.

    Attributes:
        explanation_type (str):
        processing_options (Union['ExplanationVisualizationDataProcessingOptionsType0', None, Unset]):
        outputs (Union['ExplanationVisualizationDataOutputsType0', None, Unset]):
    """

    explanation_type: str
    processing_options: Union["ExplanationVisualizationDataProcessingOptionsType0", None, Unset] = UNSET
    outputs: Union["ExplanationVisualizationDataOutputsType0", None, Unset] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.explanation_visualization_data_outputs_type_0 import ExplanationVisualizationDataOutputsType0
        from ..models.explanation_visualization_data_processing_options_type_0 import (
            ExplanationVisualizationDataProcessingOptionsType0,
        )

        explanation_type = self.explanation_type

        processing_options: Union[None, Unset, dict[str, Any]]
        if isinstance(self.processing_options, Unset):
            processing_options = UNSET
        elif isinstance(self.processing_options, ExplanationVisualizationDataProcessingOptionsType0):
            processing_options = self.processing_options.to_dict()
        else:
            processing_options = self.processing_options

        outputs: Union[None, Unset, dict[str, Any]]
        if isinstance(self.outputs, Unset):
            outputs = UNSET
        elif isinstance(self.outputs, ExplanationVisualizationDataOutputsType0):
            outputs = self.outputs.to_dict()
        else:
            outputs = self.outputs

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "explanation_type": explanation_type,
            }
        )
        if processing_options is not UNSET:
            field_dict["processing_options"] = processing_options
        if outputs is not UNSET:
            field_dict["outputs"] = outputs

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.explanation_visualization_data_outputs_type_0 import ExplanationVisualizationDataOutputsType0
        from ..models.explanation_visualization_data_processing_options_type_0 import (
            ExplanationVisualizationDataProcessingOptionsType0,
        )

        d = dict(src_dict)
        explanation_type = d.pop("explanation_type")

        def _parse_processing_options(
            data: object,
        ) -> Union["ExplanationVisualizationDataProcessingOptionsType0", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                processing_options_type_0 = ExplanationVisualizationDataProcessingOptionsType0.from_dict(data)

                return processing_options_type_0
            except:  # noqa: E722
                pass
            return cast(Union["ExplanationVisualizationDataProcessingOptionsType0", None, Unset], data)

        processing_options = _parse_processing_options(d.pop("processing_options", UNSET))

        def _parse_outputs(data: object) -> Union["ExplanationVisualizationDataOutputsType0", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                outputs_type_0 = ExplanationVisualizationDataOutputsType0.from_dict(data)

                return outputs_type_0
            except:  # noqa: E722
                pass
            return cast(Union["ExplanationVisualizationDataOutputsType0", None, Unset], data)

        outputs = _parse_outputs(d.pop("outputs", UNSET))

        explanation_visualization_data = cls(
            explanation_type=explanation_type,
            processing_options=processing_options,
            outputs=outputs,
        )

        explanation_visualization_data.additional_properties = d
        return explanation_visualization_data

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
