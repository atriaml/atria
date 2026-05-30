from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.dataset_storage_metadata_splits import DatasetStorageMetadataSplits
    from ..models.lake_fs_branch_summary import LakeFSBranchSummary


T = TypeVar("T", bound="DatasetStorageMetadata")


@_attrs_define
class DatasetStorageMetadata:
    """
    Attributes:
        main_branch (str):
        branches (list[LakeFSBranchSummary]):
        splits (DatasetStorageMetadataSplits | Unset):
        status (None | str | Unset):
    """

    main_branch: str
    branches: list[LakeFSBranchSummary]
    splits: DatasetStorageMetadataSplits | Unset = UNSET
    status: None | str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        main_branch = self.main_branch

        branches = []
        for branches_item_data in self.branches:
            branches_item = branches_item_data.to_dict()
            branches.append(branches_item)

        splits: dict[str, Any] | Unset = UNSET
        if not isinstance(self.splits, Unset):
            splits = self.splits.to_dict()

        status: None | str | Unset
        if isinstance(self.status, Unset):
            status = UNSET
        else:
            status = self.status

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "main_branch": main_branch,
                "branches": branches,
            }
        )
        if splits is not UNSET:
            field_dict["splits"] = splits
        if status is not UNSET:
            field_dict["status"] = status

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.dataset_storage_metadata_splits import DatasetStorageMetadataSplits
        from ..models.lake_fs_branch_summary import LakeFSBranchSummary

        d = dict(src_dict)
        main_branch = d.pop("main_branch")

        branches = []
        _branches = d.pop("branches")
        for branches_item_data in _branches:
            branches_item = LakeFSBranchSummary.from_dict(branches_item_data)

            branches.append(branches_item)

        _splits = d.pop("splits", UNSET)
        splits: DatasetStorageMetadataSplits | Unset
        if isinstance(_splits, Unset):
            splits = UNSET
        else:
            splits = DatasetStorageMetadataSplits.from_dict(_splits)

        def _parse_status(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        status = _parse_status(d.pop("status", UNSET))

        dataset_storage_metadata = cls(
            main_branch=main_branch,
            branches=branches,
            splits=splits,
            status=status,
        )

        dataset_storage_metadata.additional_properties = d
        return dataset_storage_metadata

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
