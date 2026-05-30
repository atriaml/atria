from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="FinalizeDatasetTaskConfig")


@_attrs_define
class FinalizeDatasetTaskConfig:
    """Config for uploading raw images/PDFs and converting to DeltaLake.

    Attributes:
        dataset_id (str):
        branch (Union[Unset, str]):  Default: 'main'.
        config_name (Union[Unset, str]):  Default: 'default'.
    """

    dataset_id: str
    branch: Unset | str = "main"
    config_name: Unset | str = "default"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        dataset_id = self.dataset_id

        branch = self.branch

        config_name = self.config_name

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "dataset_id": dataset_id,
            }
        )
        if branch is not UNSET:
            field_dict["branch"] = branch
        if config_name is not UNSET:
            field_dict["config_name"] = config_name

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        dataset_id = d.pop("dataset_id")

        branch = d.pop("branch", UNSET)

        config_name = d.pop("config_name", UNSET)

        finalize_dataset_task_config = cls(
            dataset_id=dataset_id,
            branch=branch,
            config_name=config_name,
        )

        finalize_dataset_task_config.additional_properties = d
        return finalize_dataset_task_config

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
