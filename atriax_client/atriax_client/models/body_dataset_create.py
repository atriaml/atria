from __future__ import annotations

from collections.abc import Mapping
from io import BytesIO
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from .. import types
from ..models.data_instance_type import DataInstanceType
from ..types import UNSET, File, FileTypes, Unset

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
        files (list[File] | None | Unset):
    """

    name: str
    data_instance_type: DataInstanceType
    default_branch: str | Unset = "main"
    description: str | Unset = (
        "A short description of the dataset, its intended use, and any other relevant information."
    )
    is_public: bool | Unset = False
    files: list[File] | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        data_instance_type = self.data_instance_type.value

        default_branch = self.default_branch

        description = self.description

        is_public = self.is_public

        files: list[FileTypes] | None | Unset
        if isinstance(self.files, Unset):
            files = UNSET
        elif isinstance(self.files, list):
            files = []
            for files_type_0_item_data in self.files:
                files_type_0_item = files_type_0_item_data.to_tuple()

                files.append(files_type_0_item)

        else:
            files = self.files

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
        if files is not UNSET:
            field_dict["files"] = files

        return field_dict

    def to_multipart(self) -> types.RequestFiles:
        files: types.RequestFiles = []

        files.append(("name", (None, str(self.name).encode(), "text/plain")))

        files.append(("data_instance_type", (None, str(self.data_instance_type.value).encode(), "text/plain")))

        if not isinstance(self.default_branch, Unset):
            files.append(("default_branch", (None, str(self.default_branch).encode(), "text/plain")))

        if not isinstance(self.description, Unset):
            files.append(("description", (None, str(self.description).encode(), "text/plain")))

        if not isinstance(self.is_public, Unset):
            files.append(("is_public", (None, str(self.is_public).encode(), "text/plain")))

        if not isinstance(self.files, Unset):
            if isinstance(self.files, list):
                for files_type_0_item_element in self.files:
                    files.append(("files", files_type_0_item_element.to_tuple()))
            else:
                files.append(("files", (None, str(self.files).encode(), "text/plain")))

        for prop_name, prop in self.additional_properties.items():
            files.append((prop_name, (None, str(prop).encode(), "text/plain")))

        return files

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        name = d.pop("name")

        data_instance_type = DataInstanceType(d.pop("data_instance_type"))

        default_branch = d.pop("default_branch", UNSET)

        description = d.pop("description", UNSET)

        is_public = d.pop("is_public", UNSET)

        def _parse_files(data: object) -> list[File] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                files_type_0 = []
                _files_type_0 = data
                for files_type_0_item_data in _files_type_0:
                    files_type_0_item = File(payload=BytesIO(files_type_0_item_data))

                    files_type_0.append(files_type_0_item)

                return files_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[File] | None | Unset, data)

        files = _parse_files(d.pop("files", UNSET))

        body_dataset_create = cls(
            name=name,
            data_instance_type=data_instance_type,
            default_branch=default_branch,
            description=description,
            is_public=is_public,
            files=files,
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
