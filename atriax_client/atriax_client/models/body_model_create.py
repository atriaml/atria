from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.task_type import TaskType
from ..types import UNSET, Unset

T = TypeVar("T", bound="BodyModelCreate")


@_attrs_define
class BodyModelCreate:
    """
    Attributes:
        name (str):
        task_type (TaskType):
        default_branch (str | Unset):  Default: 'main'.
        description (str | Unset):
        is_public (bool | Unset):  Default: False.
    """

    name: str
    task_type: TaskType
    default_branch: str | Unset = "main"
    description: str | Unset = UNSET
    is_public: bool | Unset = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        task_type = self.task_type.value

        default_branch = self.default_branch

        description = self.description

        is_public = self.is_public

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "task_type": task_type,
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

        task_type = TaskType(d.pop("task_type"))

        default_branch = d.pop("default_branch", UNSET)

        description = d.pop("description", UNSET)

        is_public = d.pop("is_public", UNSET)

        body_model_create = cls(
            name=name,
            task_type=task_type,
            default_branch=default_branch,
            description=description,
            is_public=is_public,
        )

        body_model_create.additional_properties = d
        return body_model_create

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
