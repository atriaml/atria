from collections.abc import Mapping
from io import BytesIO
from typing import Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from .. import types
from ..types import UNSET, File, FileTypes, Unset

T = TypeVar("T", bound="BodyUserProfileUpdate")


@_attrs_define
class BodyUserProfileUpdate:
    """
    Attributes:
        full_name (Union[None, Unset, str]):
        bio (Union[None, Unset, str]):
        location (Union[None, Unset, str]):
        website (Union[None, Unset, str]):
        avatar_file (Union[File, None, Unset]):
    """

    full_name: Union[None, Unset, str] = UNSET
    bio: Union[None, Unset, str] = UNSET
    location: Union[None, Unset, str] = UNSET
    website: Union[None, Unset, str] = UNSET
    avatar_file: Union[File, None, Unset] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        full_name: Union[None, Unset, str]
        if isinstance(self.full_name, Unset):
            full_name = UNSET
        else:
            full_name = self.full_name

        bio: Union[None, Unset, str]
        if isinstance(self.bio, Unset):
            bio = UNSET
        else:
            bio = self.bio

        location: Union[None, Unset, str]
        if isinstance(self.location, Unset):
            location = UNSET
        else:
            location = self.location

        website: Union[None, Unset, str]
        if isinstance(self.website, Unset):
            website = UNSET
        else:
            website = self.website

        avatar_file: Union[FileTypes, None, Unset]
        if isinstance(self.avatar_file, Unset):
            avatar_file = UNSET
        elif isinstance(self.avatar_file, File):
            avatar_file = self.avatar_file.to_tuple()

        else:
            avatar_file = self.avatar_file

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if full_name is not UNSET:
            field_dict["full_name"] = full_name
        if bio is not UNSET:
            field_dict["bio"] = bio
        if location is not UNSET:
            field_dict["location"] = location
        if website is not UNSET:
            field_dict["website"] = website
        if avatar_file is not UNSET:
            field_dict["avatar_file"] = avatar_file

        return field_dict

    def to_multipart(self) -> types.RequestFiles:
        files: types.RequestFiles = []

        if not isinstance(self.full_name, Unset):
            if isinstance(self.full_name, str):
                files.append(("full_name", (None, str(self.full_name).encode(), "text/plain")))
            else:
                files.append(("full_name", (None, str(self.full_name).encode(), "text/plain")))

        if not isinstance(self.bio, Unset):
            if isinstance(self.bio, str):
                files.append(("bio", (None, str(self.bio).encode(), "text/plain")))
            else:
                files.append(("bio", (None, str(self.bio).encode(), "text/plain")))

        if not isinstance(self.location, Unset):
            if isinstance(self.location, str):
                files.append(("location", (None, str(self.location).encode(), "text/plain")))
            else:
                files.append(("location", (None, str(self.location).encode(), "text/plain")))

        if not isinstance(self.website, Unset):
            if isinstance(self.website, str):
                files.append(("website", (None, str(self.website).encode(), "text/plain")))
            else:
                files.append(("website", (None, str(self.website).encode(), "text/plain")))

        if not isinstance(self.avatar_file, Unset):
            if isinstance(self.avatar_file, File):
                files.append(("avatar_file", self.avatar_file.to_tuple()))
            else:
                files.append(("avatar_file", (None, str(self.avatar_file).encode(), "text/plain")))

        for prop_name, prop in self.additional_properties.items():
            files.append((prop_name, (None, str(prop).encode(), "text/plain")))

        return files

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)

        def _parse_full_name(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        full_name = _parse_full_name(d.pop("full_name", UNSET))

        def _parse_bio(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        bio = _parse_bio(d.pop("bio", UNSET))

        def _parse_location(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        location = _parse_location(d.pop("location", UNSET))

        def _parse_website(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        website = _parse_website(d.pop("website", UNSET))

        def _parse_avatar_file(data: object) -> Union[File, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, bytes):
                    raise TypeError()
                avatar_file_type_0 = File(payload=BytesIO(data))

                return avatar_file_type_0
            except:  # noqa: E722
                pass
            return cast(Union[File, None, Unset], data)

        avatar_file = _parse_avatar_file(d.pop("avatar_file", UNSET))

        body_user_profile_update = cls(
            full_name=full_name,
            bio=bio,
            location=location,
            website=website,
            avatar_file=avatar_file,
        )

        body_user_profile_update.additional_properties = d
        return body_user_profile_update

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
