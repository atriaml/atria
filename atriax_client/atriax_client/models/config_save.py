from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.config_type import ConfigType
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.config_save_children_type_0 import ConfigSaveChildrenType0
    from ..models.config_save_params_type_0 import ConfigSaveParamsType0


T = TypeVar("T", bound="ConfigSave")


@_attrs_define
class ConfigSave:
    """
    Attributes:
        config_type (ConfigType):
        name (str):
        variant (str):
        params (ConfigSaveParamsType0 | None | Unset):
        children (ConfigSaveChildrenType0 | None | Unset):
    """

    config_type: ConfigType
    name: str
    variant: str
    params: ConfigSaveParamsType0 | None | Unset = UNSET
    children: ConfigSaveChildrenType0 | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.config_save_children_type_0 import ConfigSaveChildrenType0
        from ..models.config_save_params_type_0 import ConfigSaveParamsType0

        config_type = self.config_type.value

        name = self.name

        variant = self.variant

        params: dict[str, Any] | None | Unset
        if isinstance(self.params, Unset):
            params = UNSET
        elif isinstance(self.params, ConfigSaveParamsType0):
            params = self.params.to_dict()
        else:
            params = self.params

        children: dict[str, Any] | None | Unset
        if isinstance(self.children, Unset):
            children = UNSET
        elif isinstance(self.children, ConfigSaveChildrenType0):
            children = self.children.to_dict()
        else:
            children = self.children

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "config_type": config_type,
                "name": name,
                "variant": variant,
            }
        )
        if params is not UNSET:
            field_dict["params"] = params
        if children is not UNSET:
            field_dict["children"] = children

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.config_save_children_type_0 import ConfigSaveChildrenType0
        from ..models.config_save_params_type_0 import ConfigSaveParamsType0

        d = dict(src_dict)
        config_type = ConfigType(d.pop("config_type"))

        name = d.pop("name")

        variant = d.pop("variant")

        def _parse_params(data: object) -> ConfigSaveParamsType0 | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                params_type_0 = ConfigSaveParamsType0.from_dict(data)

                return params_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(ConfigSaveParamsType0 | None | Unset, data)

        params = _parse_params(d.pop("params", UNSET))

        def _parse_children(data: object) -> ConfigSaveChildrenType0 | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                children_type_0 = ConfigSaveChildrenType0.from_dict(data)

                return children_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(ConfigSaveChildrenType0 | None | Unset, data)

        children = _parse_children(d.pop("children", UNSET))

        config_save = cls(
            config_type=config_type,
            name=name,
            variant=variant,
            params=params,
            children=children,
        )

        config_save.additional_properties = d
        return config_save

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
