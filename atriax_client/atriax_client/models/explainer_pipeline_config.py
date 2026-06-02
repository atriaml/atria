from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.explainer_pipeline_config_params_type_0 import ExplainerPipelineConfigParamsType0


T = TypeVar("T", bound="ExplainerPipelineConfig")


@_attrs_define
class ExplainerPipelineConfig:
    """
    Attributes:
        config_schema (str):
        params (ExplainerPipelineConfigParamsType0 | None | Unset):
    """

    config_schema: str
    params: ExplainerPipelineConfigParamsType0 | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.explainer_pipeline_config_params_type_0 import ExplainerPipelineConfigParamsType0

        config_schema = self.config_schema

        params: dict[str, Any] | None | Unset
        if isinstance(self.params, Unset):
            params = UNSET
        elif isinstance(self.params, ExplainerPipelineConfigParamsType0):
            params = self.params.to_dict()
        else:
            params = self.params

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "config_schema": config_schema,
            }
        )
        if params is not UNSET:
            field_dict["params"] = params

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.explainer_pipeline_config_params_type_0 import ExplainerPipelineConfigParamsType0

        d = dict(src_dict)
        config_schema = d.pop("config_schema")

        def _parse_params(data: object) -> ExplainerPipelineConfigParamsType0 | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                params_type_0 = ExplainerPipelineConfigParamsType0.from_dict(data)

                return params_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(ExplainerPipelineConfigParamsType0 | None | Unset, data)

        params = _parse_params(d.pop("params", UNSET))

        explainer_pipeline_config = cls(
            config_schema=config_schema,
            params=params,
        )

        explainer_pipeline_config.additional_properties = d
        return explainer_pipeline_config

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
