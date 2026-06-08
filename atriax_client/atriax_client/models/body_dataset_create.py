from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

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
        data_instance_type (DataInstanceType):
        default_branch (str | Unset):  Default: 'main'.
        description (str | Unset):  Default: 'A short description of the dataset, its intended use, and any other
            relevant information.'.
        is_public (bool | Unset):  Default: False.
    """

    name: str
    data_instance_type: DataInstanceType
    default_branch: str | Unset = "main"
    description: str | Unset = (
        "A short description of the dataset, its intended use, and any other relevant information."
    )
    is_public: bool | Unset = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        data_instance_type = self.data_instance_type.value

        default_branch = self.default_branch

        description = self.description

        is_public = self.is_public

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "data_instance_type": data_instance_type,
            }
        )
        if default_branch is not UNSET:
            field_dict["default_branch"] = default_branch
        if description is not UNSET:
            field_dict["description"] = description
        if is_public is not UNSET:
            field_dict["is_public"] = is_public

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        name = d.pop("name")

        data_instance_type = DataInstanceType(d.pop("data_instance_type"))

        default_branch = d.pop("default_branch", UNSET)

        description = d.pop("description", UNSET)

        is_public = d.pop("is_public", UNSET)

        body_dataset_create = cls(
            name=name,
            data_instance_type=data_instance_type,
            default_branch=default_branch,
            description=description,
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
