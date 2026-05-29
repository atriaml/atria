from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="BodyDatasetGenerateUploadUrls")


@_attrs_define
class BodyDatasetGenerateUploadUrls:
    """
    Attributes:
        paths (list[str]):
        content_types (list[str]):
    """

    paths: list[str]
    content_types: list[str]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        paths = self.paths

        content_types = self.content_types

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "paths": paths,
                "content_types": content_types,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        paths = cast(list[str], d.pop("paths"))

        content_types = cast(list[str], d.pop("content_types"))

        body_dataset_generate_upload_urls = cls(
            paths=paths,
            content_types=content_types,
        )

        body_dataset_generate_upload_urls.additional_properties = d
        return body_dataset_generate_upload_urls

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
