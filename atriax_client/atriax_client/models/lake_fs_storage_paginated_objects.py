from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.lake_fs_storage_object import LakeFSStorageObject


T = TypeVar("T", bound="LakeFSStoragePaginatedObjects")


@_attrs_define
class LakeFSStoragePaginatedObjects:
    """
    Attributes:
        objects (list['LakeFSStorageObject']):
        next_after (Union[None, Unset, str]):
    """

    objects: list["LakeFSStorageObject"]
    next_after: Union[None, Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        objects = []
        for objects_item_data in self.objects:
            objects_item = objects_item_data.to_dict()
            objects.append(objects_item)

        next_after: Union[None, Unset, str]
        if isinstance(self.next_after, Unset):
            next_after = UNSET
        else:
            next_after = self.next_after

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "objects": objects,
            }
        )
        if next_after is not UNSET:
            field_dict["next_after"] = next_after

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.lake_fs_storage_object import LakeFSStorageObject

        d = dict(src_dict)
        objects = []
        _objects = d.pop("objects")
        for objects_item_data in _objects:
            objects_item = LakeFSStorageObject.from_dict(objects_item_data)

            objects.append(objects_item)

        def _parse_next_after(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        next_after = _parse_next_after(d.pop("next_after", UNSET))

        lake_fs_storage_paginated_objects = cls(
            objects=objects,
            next_after=next_after,
        )

        lake_fs_storage_paginated_objects.additional_properties = d
        return lake_fs_storage_paginated_objects

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
