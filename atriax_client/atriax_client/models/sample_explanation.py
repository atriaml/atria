from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.config_base import ConfigBase
    from ..models.sample_explanation_explanation_metadata import SampleExplanationExplanationMetadata


T = TypeVar("T", bound="SampleExplanation")


@_attrs_define
class SampleExplanation:
    """
    Attributes:
        id (UUID):
        created_at (str):
        updated_at (str):
        sample_index (int):
        name (str):
        config (ConfigBase):
        config_hash (str):
        explanation_metadata (SampleExplanationExplanationMetadata):
        evaluation_experiment_id (UUID):
        explanation_url (str):
    """

    id: UUID
    created_at: str
    updated_at: str
    sample_index: int
    name: str
    config: "ConfigBase"
    config_hash: str
    explanation_metadata: "SampleExplanationExplanationMetadata"
    evaluation_experiment_id: UUID
    explanation_url: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = str(self.id)

        created_at = self.created_at

        updated_at = self.updated_at

        sample_index = self.sample_index

        name = self.name

        config = self.config.to_dict()

        config_hash = self.config_hash

        explanation_metadata = self.explanation_metadata.to_dict()

        evaluation_experiment_id = str(self.evaluation_experiment_id)

        explanation_url = self.explanation_url

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "created_at": created_at,
                "updated_at": updated_at,
                "sample_index": sample_index,
                "name": name,
                "config": config,
                "config_hash": config_hash,
                "explanation_metadata": explanation_metadata,
                "evaluation_experiment_id": evaluation_experiment_id,
                "explanation_url": explanation_url,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.config_base import ConfigBase
        from ..models.sample_explanation_explanation_metadata import SampleExplanationExplanationMetadata

        d = dict(src_dict)
        id = UUID(d.pop("id"))

        created_at = d.pop("created_at")

        updated_at = d.pop("updated_at")

        sample_index = d.pop("sample_index")

        name = d.pop("name")

        config = ConfigBase.from_dict(d.pop("config"))

        config_hash = d.pop("config_hash")

        explanation_metadata = SampleExplanationExplanationMetadata.from_dict(d.pop("explanation_metadata"))

        evaluation_experiment_id = UUID(d.pop("evaluation_experiment_id"))

        explanation_url = d.pop("explanation_url")

        sample_explanation = cls(
            id=id,
            created_at=created_at,
            updated_at=updated_at,
            sample_index=sample_index,
            name=name,
            config=config,
            config_hash=config_hash,
            explanation_metadata=explanation_metadata,
            evaluation_experiment_id=evaluation_experiment_id,
            explanation_url=explanation_url,
        )

        sample_explanation.additional_properties = d
        return sample_explanation

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
