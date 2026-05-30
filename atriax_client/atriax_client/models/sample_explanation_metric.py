from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.config_base import ConfigBase
    from ..models.sample_explanation_metric_data import SampleExplanationMetricData


T = TypeVar("T", bound="SampleExplanationMetric")


@_attrs_define
class SampleExplanationMetric:
    """
    Attributes:
        id (UUID):
        created_at (str):
        updated_at (str):
        name (str):
        config (ConfigBase):
        config_hash (str):
        data (SampleExplanationMetricData):
        sample_explanation_id (UUID):
    """

    id: UUID
    created_at: str
    updated_at: str
    name: str
    config: ConfigBase
    config_hash: str
    data: SampleExplanationMetricData
    sample_explanation_id: UUID
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = str(self.id)

        created_at = self.created_at

        updated_at = self.updated_at

        name = self.name

        config = self.config.to_dict()

        config_hash = self.config_hash

        data = self.data.to_dict()

        sample_explanation_id = str(self.sample_explanation_id)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "created_at": created_at,
                "updated_at": updated_at,
                "name": name,
                "config": config,
                "config_hash": config_hash,
                "data": data,
                "sample_explanation_id": sample_explanation_id,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.config_base import ConfigBase
        from ..models.sample_explanation_metric_data import SampleExplanationMetricData

        d = dict(src_dict)
        id = UUID(d.pop("id"))

        created_at = d.pop("created_at")

        updated_at = d.pop("updated_at")

        name = d.pop("name")

        config = ConfigBase.from_dict(d.pop("config"))

        config_hash = d.pop("config_hash")

        data = SampleExplanationMetricData.from_dict(d.pop("data"))

        sample_explanation_id = UUID(d.pop("sample_explanation_id"))

        sample_explanation_metric = cls(
            id=id,
            created_at=created_at,
            updated_at=updated_at,
            name=name,
            config=config,
            config_hash=config_hash,
            data=data,
            sample_explanation_id=sample_explanation_id,
        )

        sample_explanation_metric.additional_properties = d
        return sample_explanation_metric

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
