from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="EvaluationExperimentCreate")


@_attrs_define
class EvaluationExperimentCreate:
    """
    Attributes:
        dataset_id (UUID):
        dataset_branch (str):
        dataset_split (str):
        model_id (UUID):
        model_branch (str):
    """

    dataset_id: UUID
    dataset_branch: str
    dataset_split: str
    model_id: UUID
    model_branch: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        dataset_id = str(self.dataset_id)

        dataset_branch = self.dataset_branch

        dataset_split = self.dataset_split

        model_id = str(self.model_id)

        model_branch = self.model_branch

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "dataset_id": dataset_id,
                "dataset_branch": dataset_branch,
                "dataset_split": dataset_split,
                "model_id": model_id,
                "model_branch": model_branch,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        dataset_id = UUID(d.pop("dataset_id"))

        dataset_branch = d.pop("dataset_branch")

        dataset_split = d.pop("dataset_split")

        model_id = UUID(d.pop("model_id"))

        model_branch = d.pop("model_branch")

        evaluation_experiment_create = cls(
            dataset_id=dataset_id,
            dataset_branch=dataset_branch,
            dataset_split=dataset_split,
            model_id=model_id,
            model_branch=model_branch,
        )

        evaluation_experiment_create.additional_properties = d
        return evaluation_experiment_create

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
