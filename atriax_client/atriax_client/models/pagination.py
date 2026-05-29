from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="Pagination")


@_attrs_define
class Pagination:
    """
    Attributes:
        has_more (bool): Next page is available
        next_offset (str): Token used to retrieve the next page
        results (int): Number of values found in the results
        max_per_page (int): Maximal number of entries per page
    """

    has_more: bool
    next_offset: str
    results: int
    max_per_page: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        has_more = self.has_more

        next_offset = self.next_offset

        results = self.results

        max_per_page = self.max_per_page

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "has_more": has_more,
                "next_offset": next_offset,
                "results": results,
                "max_per_page": max_per_page,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        has_more = d.pop("has_more")

        next_offset = d.pop("next_offset")

        results = d.pop("results")

        max_per_page = d.pop("max_per_page")

        pagination = cls(
            has_more=has_more,
            next_offset=next_offset,
            results=results,
            max_per_page=max_per_page,
        )

        pagination.additional_properties = d
        return pagination

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
