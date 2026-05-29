from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union, cast
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.model_config_override_config_type_0 import ModelConfigOverrideConfigType0


T = TypeVar("T", bound="ModelConfig")


@_attrs_define
class ModelConfig:
    """
    Attributes:
        id (UUID):
        config_name (str):
        branch (str):
        override_config (Union['ModelConfigOverrideConfigType0', None, Unset]):
    """

    id: UUID
    config_name: str
    branch: str
    override_config: Union["ModelConfigOverrideConfigType0", None, Unset] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.model_config_override_config_type_0 import ModelConfigOverrideConfigType0

        id = str(self.id)

        config_name = self.config_name

        branch = self.branch

        override_config: Union[None, Unset, dict[str, Any]]
        if isinstance(self.override_config, Unset):
            override_config = UNSET
        elif isinstance(self.override_config, ModelConfigOverrideConfigType0):
            override_config = self.override_config.to_dict()
        else:
            override_config = self.override_config

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "config_name": config_name,
                "branch": branch,
            }
        )
        if override_config is not UNSET:
            field_dict["override_config"] = override_config

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.model_config_override_config_type_0 import ModelConfigOverrideConfigType0

        d = dict(src_dict)
        id = UUID(d.pop("id"))

        config_name = d.pop("config_name")

        branch = d.pop("branch")

        def _parse_override_config(data: object) -> Union["ModelConfigOverrideConfigType0", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                override_config_type_0 = ModelConfigOverrideConfigType0.from_dict(data)

                return override_config_type_0
            except:  # noqa: E722
                pass
            return cast(Union["ModelConfigOverrideConfigType0", None, Unset], data)

        override_config = _parse_override_config(d.pop("override_config", UNSET))

        model_config = cls(
            id=id,
            config_name=config_name,
            branch=branch,
            override_config=override_config,
        )

        model_config.additional_properties = d
        return model_config

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
