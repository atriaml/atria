from collections.abc import Mapping
from io import BytesIO
from typing import Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from .. import types
from ..types import UNSET, File, Unset

T = TypeVar("T", bound="BodyDatasetUpload")


@_attrs_define
class BodyDatasetUpload:
    """
    Attributes:
        files (list[File]):
        paths (list[str]):
        split (Union[Unset, str]):  Default: 'train'.
        config (Union[Unset, str]):  Default: 'default'.
        validate_dataset (Union[Unset, bool]):  Default: False.
    """

    files: list[File]
    paths: list[str]
    split: Union[Unset, str] = "train"
    config: Union[Unset, str] = "default"
    validate_dataset: Union[Unset, bool] = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        files = []
        for files_item_data in self.files:
            files_item = files_item_data.to_tuple()

            files.append(files_item)

        paths = self.paths

        split = self.split

        config = self.config

        validate_dataset = self.validate_dataset

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "files": files,
                "paths": paths,
            }
        )
        if split is not UNSET:
            field_dict["split"] = split
        if config is not UNSET:
            field_dict["config"] = config
        if validate_dataset is not UNSET:
            field_dict["validate_dataset"] = validate_dataset

        return field_dict

    def to_multipart(self) -> types.RequestFiles:
        files: types.RequestFiles = []

        for files_item_element in self.files:
            files.append(("files", files_item_element.to_tuple()))

        for paths_item_element in self.paths:
            files.append(("paths", (None, str(paths_item_element).encode(), "text/plain")))

        if not isinstance(self.split, Unset):
            files.append(("split", (None, str(self.split).encode(), "text/plain")))

        if not isinstance(self.config, Unset):
            files.append(("config", (None, str(self.config).encode(), "text/plain")))

        if not isinstance(self.validate_dataset, Unset):
            files.append(("validate_dataset", (None, str(self.validate_dataset).encode(), "text/plain")))

        for prop_name, prop in self.additional_properties.items():
            files.append((prop_name, (None, str(prop).encode(), "text/plain")))

        return files

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        files = []
        _files = d.pop("files")
        for files_item_data in _files:
            files_item = File(payload=BytesIO(files_item_data))

            files.append(files_item)

        paths = cast(list[str], d.pop("paths"))

        split = d.pop("split", UNSET)

        config = d.pop("config", UNSET)

        validate_dataset = d.pop("validate_dataset", UNSET)

        body_dataset_upload = cls(
            files=files,
            paths=paths,
            split=split,
            config=config,
            validate_dataset=validate_dataset,
        )

        body_dataset_upload.additional_properties = d
        return body_dataset_upload

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
