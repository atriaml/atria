from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="DatasetValidationConfig")


@_attrs_define
class DatasetValidationConfig:
    """Config for uploading raw images/PDFs and converting to DeltaLake.

    Attributes:
        dataset_id (str):
        files (list[str]):
        data_instance_type (str):
        branch (str | Unset):  Default: 'main'.
        split (str | Unset):  Default: 'train'.
        config_name (str | Unset):  Default: 'default'.
    """

    dataset_id: str
    files: list[str]
    data_instance_type: str
    branch: str | Unset = "main"
    split: str | Unset = "train"
    config_name: str | Unset = "default"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        dataset_id = self.dataset_id

        files = self.files

        data_instance_type = self.data_instance_type

        branch = self.branch

        split = self.split

        config_name = self.config_name

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "dataset_id": dataset_id,
                "files": files,
                "data_instance_type": data_instance_type,
            }
        )
        if branch is not UNSET:
            field_dict["branch"] = branch
        if split is not UNSET:
            field_dict["split"] = split
        if config_name is not UNSET:
            field_dict["config_name"] = config_name

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        dataset_id = d.pop("dataset_id")

        files = cast(list[str], d.pop("files"))

        data_instance_type = d.pop("data_instance_type")

        branch = d.pop("branch", UNSET)

        split = d.pop("split", UNSET)

        config_name = d.pop("config_name", UNSET)

        dataset_validation_config = cls(
            dataset_id=dataset_id,
            files=files,
            data_instance_type=data_instance_type,
            branch=branch,
            split=split,
            config_name=config_name,
        )

        dataset_validation_config.additional_properties = d
        return dataset_validation_config

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
