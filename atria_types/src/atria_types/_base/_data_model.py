from typing import Self

from atria_logger import get_logger
from pydantic import BaseModel, ConfigDict

from atria_types._base._mixins._file_path_convertible import FilePathConvertible
from atria_types._base._mixins._loadable import Loadable
from atria_types._base._mixins._table_serializable import TableSerializable
from atria_types._utilities._repr import RepresentationMixin

logger = get_logger(__name__)


def _load_any(value):
    if isinstance(value, Loadable):
        return value.load()
    if isinstance(value, list):
        return [_load_any(v) for v in value]
    if isinstance(value, dict):
        return {k: _load_any(v) for k, v in value.items()}
    return value


class PydanticBase(RepresentationMixin, BaseModel):  # type: ignore[misc]
    pass


class BaseDataModel(  # type: ignore[misc]
    PydanticBase,
    Loadable,
    TableSerializable,
    FilePathConvertible,
):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
        strict=True,
        frozen=True,
    )

    def load(self) -> Self:
        loaded_fields = {}
        for name in self.__class__.model_fields:
            loaded_fields[name] = _load_any(getattr(self, name))
        new_instance = self.model_copy(update=loaded_fields)
        return new_instance
