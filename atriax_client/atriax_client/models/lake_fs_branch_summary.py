from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="LakeFSBranchSummary")


@_attrs_define
class LakeFSBranchSummary:
    """
    Attributes:
        name (str):
        last_commit_id (str):
        last_committer (str):
        last_commit_message (str):
        last_commit_date (int):
    """

    name: str
    last_commit_id: str
    last_committer: str
    last_commit_message: str
    last_commit_date: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        last_commit_id = self.last_commit_id

        last_committer = self.last_committer

        last_commit_message = self.last_commit_message

        last_commit_date = self.last_commit_date

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "last_commit_id": last_commit_id,
                "last_committer": last_committer,
                "last_commit_message": last_commit_message,
                "last_commit_date": last_commit_date,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        name = d.pop("name")

        last_commit_id = d.pop("last_commit_id")

        last_committer = d.pop("last_committer")

        last_commit_message = d.pop("last_commit_message")

        last_commit_date = d.pop("last_commit_date")

        lake_fs_branch_summary = cls(
            name=name,
            last_commit_id=last_commit_id,
            last_committer=last_committer,
            last_commit_message=last_commit_message,
            last_commit_date=last_commit_date,
        )

        lake_fs_branch_summary.additional_properties = d
        return lake_fs_branch_summary

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
