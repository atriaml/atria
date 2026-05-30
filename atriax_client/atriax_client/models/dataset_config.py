from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.dataset_split_type import DatasetSplitType
from ..types import UNSET, Unset

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
        sample_indices (list[int] | None | Unset):
    """

    id: UUID
    branch: str
    split: DatasetSplitType
    sample_indices: list[int] | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = str(self.id)

        branch = self.branch

        split = self.split.value

        sample_indices: list[int] | None | Unset
        if isinstance(self.sample_indices, Unset):
            sample_indices = UNSET
        elif isinstance(self.sample_indices, list):
            sample_indices = self.sample_indices

        else:
            sample_indices = self.sample_indices

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "branch": branch,
                "split": split,
            }
        )
        if sample_indices is not UNSET:
            field_dict["sample_indices"] = sample_indices

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        id = UUID(d.pop("id"))

        branch = d.pop("branch")

        split = DatasetSplitType(d.pop("split"))

        def _parse_sample_indices(data: object) -> list[int] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                sample_indices_type_0 = cast(list[int], data)

                return sample_indices_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[int] | None | Unset, data)

        sample_indices = _parse_sample_indices(d.pop("sample_indices", UNSET))

        dataset_config = cls(
            id=id,
            branch=branch,
            split=split,
            sample_indices=sample_indices,
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
