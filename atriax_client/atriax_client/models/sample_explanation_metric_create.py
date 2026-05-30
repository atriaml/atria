from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.config_base import ConfigBase
    from ..models.sample_explanation_metric_create_data import SampleExplanationMetricCreateData


T = TypeVar("T", bound="SampleExplanationMetricCreate")


@_attrs_define
class SampleExplanationMetricCreate:
    """
    Attributes:
        name (str):
        config (ConfigBase):
        data (SampleExplanationMetricCreateData):
        config_hash (Union[None, Unset, str]):
    """

    name: str
    config: "ConfigBase"
    data: "SampleExplanationMetricCreateData"
    config_hash: None | Unset | str = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        config = self.config.to_dict()

        data = self.data.to_dict()

        config_hash: None | Unset | str
        if isinstance(self.config_hash, Unset):
            config_hash = UNSET
        else:
            config_hash = self.config_hash

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "config": config,
                "data": data,
            }
        )
        if config_hash is not UNSET:
            field_dict["config_hash"] = config_hash

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.config_base import ConfigBase
        from ..models.sample_explanation_metric_create_data import SampleExplanationMetricCreateData

        d = dict(src_dict)
        name = d.pop("name")

        config = ConfigBase.from_dict(d.pop("config"))

        data = SampleExplanationMetricCreateData.from_dict(d.pop("data"))

        def _parse_config_hash(data: object) -> None | Unset | str:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | Unset | str, data)

        config_hash = _parse_config_hash(d.pop("config_hash", UNSET))

        sample_explanation_metric_create = cls(
            name=name,
            config=config,
            data=data,
            config_hash=config_hash,
        )

        sample_explanation_metric_create.additional_properties = d
        return sample_explanation_metric_create

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
