from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.pre_signed_url_response_item import PreSignedUrlResponseItem


T = TypeVar("T", bound="PreSignedUrlResponse")


@_attrs_define
class PreSignedUrlResponse:
    """
    Attributes:
        upload_urls (list['PreSignedUrlResponseItem']):
    """

    upload_urls: list["PreSignedUrlResponseItem"]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        upload_urls = []
        for upload_urls_item_data in self.upload_urls:
            upload_urls_item = upload_urls_item_data.to_dict()
            upload_urls.append(upload_urls_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "upload_urls": upload_urls,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.pre_signed_url_response_item import PreSignedUrlResponseItem

        d = dict(src_dict)
        upload_urls = []
        _upload_urls = d.pop("upload_urls")
        for upload_urls_item_data in _upload_urls:
            upload_urls_item = PreSignedUrlResponseItem.from_dict(upload_urls_item_data)

            upload_urls.append(upload_urls_item)

        pre_signed_url_response = cls(
            upload_urls=upload_urls,
        )

        pre_signed_url_response.additional_properties = d
        return pre_signed_url_response

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
