from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union, cast
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.config_type import ConfigType
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.config_children_type_0 import ConfigChildrenType0
    from ..models.config_params_type_0 import ConfigParamsType0


T = TypeVar("T", bound="Config")


@_attrs_define
class Config:
    """
    Attributes:
        id (UUID):
        created_at (str):
        updated_at (str):
        config_type (ConfigType):
        name (str):
        module_path (str):
        user_id (UUID):
        variant (Union[Unset, str]):  Default: 'default'.
        hash_ (Union[None, Unset, str]):
        hash_fields (Union[Unset, list[str]]):
        params (Union['ConfigParamsType0', None, Unset]):
        children (Union['ConfigChildrenType0', None, Unset]):
    """

    id: UUID
    created_at: str
    updated_at: str
    config_type: ConfigType
    name: str
    module_path: str
    user_id: UUID
    variant: Union[Unset, str] = "default"
    hash_: Union[None, Unset, str] = UNSET
    hash_fields: Union[Unset, list[str]] = UNSET
    params: Union["ConfigParamsType0", None, Unset] = UNSET
    children: Union["ConfigChildrenType0", None, Unset] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.config_children_type_0 import ConfigChildrenType0
        from ..models.config_params_type_0 import ConfigParamsType0

        id = str(self.id)

        created_at = self.created_at

        updated_at = self.updated_at

        config_type = self.config_type.value

        name = self.name

        module_path = self.module_path

        user_id = str(self.user_id)

        variant = self.variant

        hash_: Union[None, Unset, str]
        if isinstance(self.hash_, Unset):
            hash_ = UNSET
        else:
            hash_ = self.hash_

        hash_fields: Union[Unset, list[str]] = UNSET
        if not isinstance(self.hash_fields, Unset):
            hash_fields = self.hash_fields

        params: Union[None, Unset, dict[str, Any]]
        if isinstance(self.params, Unset):
            params = UNSET
        elif isinstance(self.params, ConfigParamsType0):
            params = self.params.to_dict()
        else:
            params = self.params

        children: Union[None, Unset, dict[str, Any]]
        if isinstance(self.children, Unset):
            children = UNSET
        elif isinstance(self.children, ConfigChildrenType0):
            children = self.children.to_dict()
        else:
            children = self.children

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "created_at": created_at,
                "updated_at": updated_at,
                "config_type": config_type,
                "name": name,
                "module_path": module_path,
                "user_id": user_id,
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
        from ..models.config_children_type_0 import ConfigChildrenType0
        from ..models.config_params_type_0 import ConfigParamsType0

        d = dict(src_dict)
        id = UUID(d.pop("id"))

        created_at = d.pop("created_at")

        updated_at = d.pop("updated_at")

        config_type = ConfigType(d.pop("config_type"))

        name = d.pop("name")

        module_path = d.pop("module_path")

        user_id = UUID(d.pop("user_id"))

        variant = d.pop("variant", UNSET)

        def _parse_hash_(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        hash_ = _parse_hash_(d.pop("hash", UNSET))

        hash_fields = cast(list[str], d.pop("hash_fields", UNSET))

        def _parse_params(data: object) -> Union["ConfigParamsType0", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                params_type_0 = ConfigParamsType0.from_dict(data)

                return params_type_0
            except:  # noqa: E722
                pass
            return cast(Union["ConfigParamsType0", None, Unset], data)

        params = _parse_params(d.pop("params", UNSET))

        def _parse_children(data: object) -> Union["ConfigChildrenType0", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                children_type_0 = ConfigChildrenType0.from_dict(data)

                return children_type_0
            except:  # noqa: E722
                pass
            return cast(Union["ConfigChildrenType0", None, Unset], data)

        children = _parse_children(d.pop("children", UNSET))

        config = cls(
            id=id,
            created_at=created_at,
            updated_at=updated_at,
            config_type=config_type,
            name=name,
            module_path=module_path,
            user_id=user_id,
            variant=variant,
            hash_=hash_,
            hash_fields=hash_fields,
            params=params,
            children=children,
        )

        config.additional_properties = d
        return config

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
