from collections.abc import Mapping
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.data_instance_type import DataInstanceType
from ..types import UNSET, Unset

T = TypeVar("T", bound="BodyDatasetCreate")


@_attrs_define
class BodyDatasetCreate:
    """
    Attributes:
        name (str):
        description (str):
        data_instance_type (DataInstanceType):
        default_branch (Union[Unset, str]):  Default: 'main'.
        is_public (Union[Unset, bool]):  Default: False.
    """

    name: str
    description: str
    data_instance_type: DataInstanceType
    default_branch: Union[Unset, str] = "main"
    is_public: Union[Unset, bool] = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        description = self.description

        data_instance_type = self.data_instance_type.value

        default_branch = self.default_branch

        is_public = self.is_public

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "description": description,
                "data_instance_type": data_instance_type,
            }
        )
        if default_branch is not UNSET:
            field_dict["default_branch"] = default_branch
        if is_public is not UNSET:
            field_dict["is_public"] = is_public

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        name = d.pop("name")

        description = d.pop("description")

        data_instance_type = DataInstanceType(d.pop("data_instance_type"))

        default_branch = d.pop("default_branch", UNSET)

        is_public = d.pop("is_public", UNSET)

        body_dataset_create = cls(
            name=name,
            description=description,
            data_instance_type=data_instance_type,
            default_branch=default_branch,
            is_public=is_public,
        )

        body_dataset_create.additional_properties = d
        return body_dataset_create

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
