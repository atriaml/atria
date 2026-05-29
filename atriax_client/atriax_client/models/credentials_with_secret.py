from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="CredentialsWithSecret")


@_attrs_define
class CredentialsWithSecret:
    """
    Attributes:
        access_key_id (str):
        secret_access_key (str):
        creation_date (int): Unix Epoch in seconds
    """

    access_key_id: str
    secret_access_key: str
    creation_date: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        access_key_id = self.access_key_id

        secret_access_key = self.secret_access_key

        creation_date = self.creation_date

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "access_key_id": access_key_id,
                "secret_access_key": secret_access_key,
                "creation_date": creation_date,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        access_key_id = d.pop("access_key_id")

        secret_access_key = d.pop("secret_access_key")

        creation_date = d.pop("creation_date")

        credentials_with_secret = cls(
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            creation_date=creation_date,
        )

        credentials_with_secret.additional_properties = d
        return credentials_with_secret

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
