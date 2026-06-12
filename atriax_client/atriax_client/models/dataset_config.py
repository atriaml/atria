from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.dataset_split_type import DatasetSplitType

T = TypeVar("T", bound="DatasetConfig")


@_attrs_define
class DatasetConfig:
    """
    Attributes:
        id (UUID):
        branch (str):
        split (DatasetSplitType): An enumeration representing the dataset splits.

            Attributes:
                train (str): Represents the training split of the dataset.
                test (str): Represents the testing split of the dataset.
                validation (str): Represents the validation split of the dataset.
    """

    id: UUID
    branch: str
    split: DatasetSplitType
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = str(self.id)

        branch = self.branch

        split = self.split.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "branch": branch,
                "split": split,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        id = UUID(d.pop("id"))

        branch = d.pop("branch")

        split = DatasetSplitType(d.pop("split"))

        dataset_config = cls(
            id=id,
            branch=branch,
            split=split,
        )

        dataset_config.additional_properties = d
        return dataset_config

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
