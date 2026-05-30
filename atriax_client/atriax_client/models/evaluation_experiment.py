from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="EvaluationExperiment")


@_attrs_define
class EvaluationExperiment:
    """
    Attributes:
        id (UUID):
        created_at (str):
        updated_at (str):
        dataset_id (UUID):
        dataset_branch_commit_sha (str):
        dataset_config_name (str):
        dataset_split (str):
        model_id (UUID):
        model_branch_commit_sha (str):
        model_config_name (str):
        user_id (UUID):
        is_public (bool | Unset):  Default: False.
    """

    id: UUID
    created_at: str
    updated_at: str
    dataset_id: UUID
    dataset_branch_commit_sha: str
    dataset_config_name: str
    dataset_split: str
    model_id: UUID
    model_branch_commit_sha: str
    model_config_name: str
    user_id: UUID
    is_public: bool | Unset = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = str(self.id)

        created_at = self.created_at

        updated_at = self.updated_at

        dataset_id = str(self.dataset_id)

        dataset_branch_commit_sha = self.dataset_branch_commit_sha

        dataset_config_name = self.dataset_config_name

        dataset_split = self.dataset_split

        model_id = str(self.model_id)

        model_branch_commit_sha = self.model_branch_commit_sha

        model_config_name = self.model_config_name

        user_id = str(self.user_id)

        is_public = self.is_public

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "created_at": created_at,
                "updated_at": updated_at,
                "dataset_id": dataset_id,
                "dataset_branch_commit_sha": dataset_branch_commit_sha,
                "dataset_config_name": dataset_config_name,
                "dataset_split": dataset_split,
                "model_id": model_id,
                "model_branch_commit_sha": model_branch_commit_sha,
                "model_config_name": model_config_name,
                "user_id": user_id,
            }
        )
        if is_public is not UNSET:
            field_dict["is_public"] = is_public

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        id = UUID(d.pop("id"))

        created_at = d.pop("created_at")

        updated_at = d.pop("updated_at")

        dataset_id = UUID(d.pop("dataset_id"))

        dataset_branch_commit_sha = d.pop("dataset_branch_commit_sha")

        dataset_config_name = d.pop("dataset_config_name")

        dataset_split = d.pop("dataset_split")

        model_id = UUID(d.pop("model_id"))

        model_branch_commit_sha = d.pop("model_branch_commit_sha")

        model_config_name = d.pop("model_config_name")

        user_id = UUID(d.pop("user_id"))

        is_public = d.pop("is_public", UNSET)

        evaluation_experiment = cls(
            id=id,
            created_at=created_at,
            updated_at=updated_at,
            dataset_id=dataset_id,
            dataset_branch_commit_sha=dataset_branch_commit_sha,
            dataset_config_name=dataset_config_name,
            dataset_split=dataset_split,
            model_id=model_id,
            model_branch_commit_sha=model_branch_commit_sha,
            model_config_name=model_config_name,
            user_id=user_id,
            is_public=is_public,
        )

        evaluation_experiment.additional_properties = d
        return evaluation_experiment

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
