from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define

if TYPE_CHECKING:
    from ..models.dataset_samples_page_items_item import DatasetSamplesPageItemsItem


T = TypeVar("T", bound="DatasetSamplesPage")


@_attrs_define
class DatasetSamplesPage:
    """
    Attributes:
        items (list['DatasetSamplesPageItemsItem']):
        total (int):
        page (int):
        page_size (int):
    """

    items: list["DatasetSamplesPageItemsItem"]
    total: int
    page: int
    page_size: int

    def to_dict(self) -> dict[str, Any]:
        items = []
        for items_item_data in self.items:
            items_item = items_item_data.to_dict()
            items.append(items_item)

        total = self.total

        page = self.page

        page_size = self.page_size

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "items": items,
                "total": total,
                "page": page,
                "page_size": page_size,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.dataset_samples_page_items_item import DatasetSamplesPageItemsItem

        d = dict(src_dict)
        items = []
        _items = d.pop("items")
        for items_item_data in _items:
            items_item = DatasetSamplesPageItemsItem.from_dict(items_item_data)

            items.append(items_item)

        total = d.pop("total")

        page = d.pop("page")

        page_size = d.pop("page_size")

        dataset_samples_page = cls(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
        )

        return dataset_samples_page
