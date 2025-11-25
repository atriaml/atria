from typing import Self

from atria_logger import get_logger
from atria_types.utilities.repr import RepresentationMixin
from pydantic import BaseModel, ConfigDict

from atria_types._base._mixins._file_path_convertible import FilePathConvertible
from atria_types._base._mixins._loadable import Loadable
from atria_types._base._mixins._table_serializable import TableSerializable

logger = get_logger(__name__)


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
        loaded_fields = {
            name: self._load(getattr(self, name))
            for name in self.__class__.model_fields
        }

        new_instance = self.model_copy(update=loaded_fields)
        return new_instance

    def _load(self, value):
        if isinstance(value, Loadable):
            return value.load()
        if isinstance(value, list):
            return [self._load(v) for v in value]
        if isinstance(value, dict):
            return {k: self._load(v) for k, v in value.items()}
        return value
