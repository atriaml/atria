from pathlib import Path
from typing import Any

from atria_logger import get_logger
from pydantic import field_validator
from rich.repr import RichReprResult

from atria_types._base._data_model import BaseDataModel
from atria_types._pydantic import (
    OptIntField,
    OptStrField,
    ValidatedPILImage,
)

logger = get_logger(__name__)


class Image(BaseDataModel):
    file_path: OptStrField = None
    content: ValidatedPILImage = None
    source_width: OptIntField = None
    source_height: OptIntField = None

    @property
    def width(self) -> int:
        return self.size[0] if self.size else None

    @property
    def height(self) -> int:
        return self.size[1] if self.size else None

    @field_validator("file_path", mode="before")
    @classmethod
    def _validate_file_path(cls, value: Any) -> str | None:
        if isinstance(value, Path):
            return str(value)
        return value

    @property
    def size(self) -> tuple[int, int] | None:
        assert self.content is not None, (
            "Image content is not loaded. Call load() first or assign content directly."
        )
        return self.content.size

    @property
    def channels(self) -> int | list[int]:
        assert self.content is not None, (
            "Image content is not loaded. Call load() first or assign content directly."
        )
        return len(self.content.getbands())

    def _load(self):
        if self.content is None:
            from atria_types.utilities.encoding import _bytes_to_image
            from atria_types.utilities.file import _load_bytes_from_uri

            if self.file_path is None:
                raise ValueError(
                    "Image file path is not set. Please set file_path before loading."
                )

            return Image(
                file_path=self.file_path,
                content=_bytes_to_image(_load_bytes_from_uri(self.file_path)),
            )
        return self

    def __rich_repr__(self) -> RichReprResult:  # type: ignore
        yield from super().__rich_repr__()
        if self.content is not None:
            yield "width", self.width
            yield "height", self.height
            yield "channels", self.channels
