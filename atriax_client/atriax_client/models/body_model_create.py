from collections.abc import Mapping
from io import BytesIO
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from .. import types
from ..models.task_type import TaskType
from ..types import UNSET, File, FileTypes, Unset

T = TypeVar("T", bound="BodyModelCreate")


@_attrs_define
class BodyModelCreate:
    """
    Attributes:
        name (str):
        task_type (Union[Unset, TaskType]):
        model_file (Union[Unset, File]):
        default_branch (Union[Unset, str]):  Default: 'main'.
        description (Union[Unset, str]):
        is_public (Union[Unset, bool]):  Default: False.
    """

    name: str
    task_type: Union[Unset, TaskType] = UNSET
    model_file: Union[Unset, File] = UNSET
    default_branch: Union[Unset, str] = "main"
    description: Union[Unset, str] = UNSET
    is_public: Union[Unset, bool] = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        task_type: Union[Unset, str] = UNSET
        if not isinstance(self.task_type, Unset):
            task_type = self.task_type.value

        model_file: Union[Unset, FileTypes] = UNSET
        if not isinstance(self.model_file, Unset):
            model_file = self.model_file.to_tuple()

        default_branch = self.default_branch

        description = self.description

        is_public = self.is_public

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
            }
        )
        if task_type is not UNSET:
            field_dict["task_type"] = task_type
        if model_file is not UNSET:
            field_dict["model_file"] = model_file
        if default_branch is not UNSET:
            field_dict["default_branch"] = default_branch
        if description is not UNSET:
            field_dict["description"] = description
        if is_public is not UNSET:
            field_dict["is_public"] = is_public

        return field_dict

    def to_multipart(self) -> types.RequestFiles:
        files: types.RequestFiles = []

        files.append(("name", (None, str(self.name).encode(), "text/plain")))

        if not isinstance(self.task_type, Unset):
            files.append(("task_type", (None, str(self.task_type.value).encode(), "text/plain")))

        if not isinstance(self.model_file, Unset):
            files.append(("model_file", self.model_file.to_tuple()))

        if not isinstance(self.default_branch, Unset):
            files.append(("default_branch", (None, str(self.default_branch).encode(), "text/plain")))

        if not isinstance(self.description, Unset):
            files.append(("description", (None, str(self.description).encode(), "text/plain")))

        if not isinstance(self.is_public, Unset):
            files.append(("is_public", (None, str(self.is_public).encode(), "text/plain")))

        for prop_name, prop in self.additional_properties.items():
            files.append((prop_name, (None, str(prop).encode(), "text/plain")))

        return files

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        name = d.pop("name")

        _task_type = d.pop("task_type", UNSET)
        task_type: Union[Unset, TaskType]
        if isinstance(_task_type, Unset):
            task_type = UNSET
        else:
            task_type = TaskType(_task_type)

        _model_file = d.pop("model_file", UNSET)
        model_file: Union[Unset, File]
        if isinstance(_model_file, Unset):
            model_file = UNSET
        else:
            model_file = File(payload=BytesIO(_model_file))

        default_branch = d.pop("default_branch", UNSET)

        description = d.pop("description", UNSET)

        is_public = d.pop("is_public", UNSET)

        body_model_create = cls(
            name=name,
            task_type=task_type,
            model_file=model_file,
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
