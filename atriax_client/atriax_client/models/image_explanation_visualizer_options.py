from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.normalization_type import NormalizationType
from ..types import UNSET, Unset

T = TypeVar("T", bound="ImageExplanationVisualizerOptions")


@_attrs_define
class ImageExplanationVisualizerOptions:
    """Options specific to image explanation processing.

    Attributes:
        normalization_type (Union[Unset, NormalizationType]):
        outlier_perc (Union[Unset, int]):  Default: 2.
    """

    normalization_type: Unset | NormalizationType = UNSET
    outlier_perc: Unset | int = 2
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        normalization_type: Unset | str = UNSET
        if not isinstance(self.normalization_type, Unset):
            normalization_type = self.normalization_type.value

        outlier_perc = self.outlier_perc

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if normalization_type is not UNSET:
            field_dict["normalization_type"] = normalization_type
        if outlier_perc is not UNSET:
            field_dict["outlier_perc"] = outlier_perc

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        _normalization_type = d.pop("normalization_type", UNSET)
        normalization_type: Unset | NormalizationType
        if isinstance(_normalization_type, Unset):
            normalization_type = UNSET
        else:
            normalization_type = NormalizationType(_normalization_type)

        outlier_perc = d.pop("outlier_perc", UNSET)

        image_explanation_visualizer_options = cls(
            normalization_type=normalization_type,
            outlier_perc=outlier_perc,
        )

        image_explanation_visualizer_options.additional_properties = d
        return image_explanation_visualizer_options

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
