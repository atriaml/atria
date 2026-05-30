from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.config_base import ConfigBase


T = TypeVar("T", bound="ConfigChildrenType0")


@_attrs_define
class ConfigChildrenType0:
    """ """

    additional_properties: dict[str, ConfigBase | list[ConfigBase]] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.config_base import ConfigBase

        field_dict: dict[str, Any] = {}
        for prop_name, prop in self.additional_properties.items():
            if isinstance(prop, ConfigBase):
                field_dict[prop_name] = prop.to_dict()
            else:
                field_dict[prop_name] = []
                for additional_property_type_1_item_data in prop:
                    additional_property_type_1_item = additional_property_type_1_item_data.to_dict()
                    field_dict[prop_name].append(additional_property_type_1_item)

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.config_base import ConfigBase

        d = dict(src_dict)
        config_children_type_0 = cls()

        additional_properties = {}
        for prop_name, prop_dict in d.items():

            def _parse_additional_property(data: object) -> ConfigBase | list[ConfigBase]:
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    additional_property_type_0 = ConfigBase.from_dict(data)

                    return additional_property_type_0
                except (TypeError, ValueError, AttributeError, KeyError):
                    pass
                if not isinstance(data, list):
                    raise TypeError()
                additional_property_type_1 = []
                _additional_property_type_1 = data
                for additional_property_type_1_item_data in _additional_property_type_1:
                    additional_property_type_1_item = ConfigBase.from_dict(additional_property_type_1_item_data)

                    additional_property_type_1.append(additional_property_type_1_item)

                return additional_property_type_1

            additional_property = _parse_additional_property(prop_dict)

            additional_properties[prop_name] = additional_property

        config_children_type_0.additional_properties = additional_properties
        return config_children_type_0

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> ConfigBase | list[ConfigBase]:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: ConfigBase | list[ConfigBase]) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
