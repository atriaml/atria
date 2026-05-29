from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.explanation_output import ExplanationOutput


T = TypeVar("T", bound="ExplanationVisualizationDataOutputsType0")


@_attrs_define
class ExplanationVisualizationDataOutputsType0:
    """ """

    additional_properties: dict[str, "ExplanationOutput"] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        field_dict: dict[str, Any] = {}
        for prop_name, prop in self.additional_properties.items():
            field_dict[prop_name] = prop.to_dict()

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.explanation_output import ExplanationOutput

        d = dict(src_dict)
        explanation_visualization_data_outputs_type_0 = cls()

        additional_properties = {}
        for prop_name, prop_dict in d.items():
            additional_property = ExplanationOutput.from_dict(prop_dict)

            additional_properties[prop_name] = additional_property

        explanation_visualization_data_outputs_type_0.additional_properties = additional_properties
        return explanation_visualization_data_outputs_type_0

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> "ExplanationOutput":
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: "ExplanationOutput") -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
