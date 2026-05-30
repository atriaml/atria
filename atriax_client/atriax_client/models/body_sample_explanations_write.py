from __future__ import annotations

from collections.abc import Mapping
from io import BytesIO
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from .. import types
from ..types import UNSET, File, FileTypes, Unset

T = TypeVar("T", bound="BodySampleExplanationsWrite")


@_attrs_define
class BodySampleExplanationsWrite:
    """
    Attributes:
        sample_index (int):
        name (str):
        config (str):
        explanation_metadata (str):
        explanation_file (File | None | Unset):
    """

    sample_index: int
    name: str
    config: str
    explanation_metadata: str
    explanation_file: File | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        sample_index = self.sample_index

        name = self.name

        config = self.config

        explanation_metadata = self.explanation_metadata

        explanation_file: FileTypes | None | Unset
        if isinstance(self.explanation_file, Unset):
            explanation_file = UNSET
        elif isinstance(self.explanation_file, File):
            explanation_file = self.explanation_file.to_tuple()

        else:
            explanation_file = self.explanation_file

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "sample_index": sample_index,
                "name": name,
                "config": config,
                "explanation_metadata": explanation_metadata,
            }
        )
        if explanation_file is not UNSET:
            field_dict["explanation_file"] = explanation_file

        return field_dict

    def to_multipart(self) -> types.RequestFiles:
        files: types.RequestFiles = []

        files.append(("sample_index", (None, str(self.sample_index).encode(), "text/plain")))

        files.append(("name", (None, str(self.name).encode(), "text/plain")))

        files.append(("config", (None, str(self.config).encode(), "text/plain")))

        files.append(("explanation_metadata", (None, str(self.explanation_metadata).encode(), "text/plain")))

        if not isinstance(self.explanation_file, Unset):
            if isinstance(self.explanation_file, File):
                files.append(("explanation_file", self.explanation_file.to_tuple()))
            else:
                files.append(("explanation_file", (None, str(self.explanation_file).encode(), "text/plain")))

        for prop_name, prop in self.additional_properties.items():
            files.append((prop_name, (None, str(prop).encode(), "text/plain")))

        return files

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        sample_index = d.pop("sample_index")

        name = d.pop("name")

        config = d.pop("config")

        explanation_metadata = d.pop("explanation_metadata")

        def _parse_explanation_file(data: object) -> File | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, bytes):
                    raise TypeError()
                explanation_file_type_0 = File(payload=BytesIO(data))

                return explanation_file_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(File | None | Unset, data)

        explanation_file = _parse_explanation_file(d.pop("explanation_file", UNSET))

        body_sample_explanations_write = cls(
            sample_index=sample_index,
            name=name,
            config=config,
            explanation_metadata=explanation_metadata,
            explanation_file=explanation_file,
        )

        body_sample_explanations_write.additional_properties = d
        return body_sample_explanations_write

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
