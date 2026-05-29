from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.sample_evaluation_data import SampleEvaluationData


T = TypeVar("T", bound="SampleEvaluation")


@_attrs_define
class SampleEvaluation:
    """
    Attributes:
        id (UUID):
        created_at (str):
        updated_at (str):
        sample_index (int):
        data (SampleEvaluationData):
        evaluation_experiment_id (UUID):
    """

    id: UUID
    created_at: str
    updated_at: str
    sample_index: int
    data: "SampleEvaluationData"
    evaluation_experiment_id: UUID
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = str(self.id)

        created_at = self.created_at

        updated_at = self.updated_at

        sample_index = self.sample_index

        data = self.data.to_dict()

        evaluation_experiment_id = str(self.evaluation_experiment_id)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "created_at": created_at,
                "updated_at": updated_at,
                "sample_index": sample_index,
                "data": data,
                "evaluation_experiment_id": evaluation_experiment_id,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.sample_evaluation_data import SampleEvaluationData

        d = dict(src_dict)
        id = UUID(d.pop("id"))

        created_at = d.pop("created_at")

        updated_at = d.pop("updated_at")

        sample_index = d.pop("sample_index")

        data = SampleEvaluationData.from_dict(d.pop("data"))

        evaluation_experiment_id = UUID(d.pop("evaluation_experiment_id"))

        sample_evaluation = cls(
            id=id,
            created_at=created_at,
            updated_at=updated_at,
            sample_index=sample_index,
            data=data,
            evaluation_experiment_id=evaluation_experiment_id,
        )

        sample_evaluation.additional_properties = d
        return sample_evaluation

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
