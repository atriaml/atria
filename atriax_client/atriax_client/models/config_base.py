from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.config_type import ConfigType
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.config_base_children_type_0 import ConfigBaseChildrenType0
    from ..models.config_base_params_type_0 import ConfigBaseParamsType0


T = TypeVar("T", bound="ConfigBase")


@_attrs_define
class ConfigBase:
    """
    Attributes:
        config_type (ConfigType):
        name (str):
        module_path (str):
        variant (Union[Unset, str]):  Default: 'default'.
        hash_ (Union[None, Unset, str]):
        hash_fields (Union[Unset, list[str]]):
        params (Union['ConfigBaseParamsType0', None, Unset]):
        children (Union['ConfigBaseChildrenType0', None, Unset]):
    """

    config_type: ConfigType
    name: str
    module_path: str
    variant: Unset | str = "default"
    hash_: None | Unset | str = UNSET
    hash_fields: Unset | list[str] = UNSET
    params: Union["ConfigBaseParamsType0", None, Unset] = UNSET
    children: Union["ConfigBaseChildrenType0", None, Unset] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.config_base_children_type_0 import ConfigBaseChildrenType0
        from ..models.config_base_params_type_0 import ConfigBaseParamsType0

        config_type = self.config_type.value

        name = self.name

        module_path = self.module_path

        variant = self.variant

        hash_: None | Unset | str
        if isinstance(self.hash_, Unset):
            hash_ = UNSET
        else:
            hash_ = self.hash_

        hash_fields: Unset | list[str] = UNSET
        if not isinstance(self.hash_fields, Unset):
            hash_fields = self.hash_fields

        params: None | Unset | dict[str, Any]
        if isinstance(self.params, Unset):
            params = UNSET
        elif isinstance(self.params, ConfigBaseParamsType0):
            params = self.params.to_dict()
        else:
            params = self.params

        children: None | Unset | dict[str, Any]
        if isinstance(self.children, Unset):
            children = UNSET
        elif isinstance(self.children, ConfigBaseChildrenType0):
            children = self.children.to_dict()
        else:
            children = self.children

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "config_type": config_type,
                "name": name,
                "module_path": module_path,
            }
        )
        if variant is not UNSET:
            field_dict["variant"] = variant
        if hash_ is not UNSET:
            field_dict["hash"] = hash_
        if hash_fields is not UNSET:
            field_dict["hash_fields"] = hash_fields
        if params is not UNSET:
            field_dict["params"] = params
        if children is not UNSET:
            field_dict["children"] = children

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.config_base_children_type_0 import ConfigBaseChildrenType0
        from ..models.config_base_params_type_0 import ConfigBaseParamsType0

        d = dict(src_dict)
        config_type = ConfigType(d.pop("config_type"))

        name = d.pop("name")

        module_path = d.pop("module_path")

        variant = d.pop("variant", UNSET)

        def _parse_hash_(data: object) -> None | Unset | str:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | Unset | str, data)

        hash_ = _parse_hash_(d.pop("hash", UNSET))

        hash_fields = cast(list[str], d.pop("hash_fields", UNSET))

        def _parse_params(data: object) -> Union["ConfigBaseParamsType0", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                params_type_0 = ConfigBaseParamsType0.from_dict(data)

                return params_type_0
            except:  # noqa: E722
                pass
            return cast(Union["ConfigBaseParamsType0", None, Unset], data)

        params = _parse_params(d.pop("params", UNSET))

        def _parse_children(data: object) -> Union["ConfigBaseChildrenType0", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                children_type_0 = ConfigBaseChildrenType0.from_dict(data)

                return children_type_0
            except:  # noqa: E722
                pass
            return cast(Union["ConfigBaseChildrenType0", None, Unset], data)

        children = _parse_children(d.pop("children", UNSET))

        config_base = cls(
            config_type=config_type,
            name=name,
            module_path=module_path,
            variant=variant,
            hash_=hash_,
            hash_fields=hash_fields,
            params=params,
            children=children,
        )

        config_base.additional_properties = d
        return config_base

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
