from typing import Self

from pydantic import BaseModel


class Loadable(BaseModel):
    def load(self) -> Self:
        loaded_fields = {}
        for field_name in self.__class__.model_fields:
            field_value = getattr(self, field_name)
            if isinstance(field_value, Loadable):
                loaded_fields[field_name] = field_value.load()
            else:
                loaded_fields[field_name] = field_value

        new_instance = self.model_copy(update=loaded_fields)
        new_instance._load()
        return new_instance

    def _load(self) -> None:
        """
        This method is intended to be overridden by subclasses to implement custom loading logic.
        It is called automatically during the ``load()`` process after a new instance is created.
        """
