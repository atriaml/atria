from collections.abc import Callable
from typing import Any, Self, TypeVar

import torch
from atria_types._utilities._repr import RepresentationMixin
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


class TensorOperations:
    def __init__(self, model: "TensorDataModel"):
        self.model = model

    @property
    def model_fields(self):
        return self.model.__class__.model_fields

    def _map_tensors(self, fn: Callable) -> "TensorDataModel":
        """Apply function to all tensor fields."""
        updates = {}
        for field_name in self.model_fields.keys():
            if field_name == "metadata":
                continue
            val = getattr(self.model, field_name)
            updates[field_name] = fn(val) if isinstance(val, torch.Tensor) else val

        return self.model.model_copy(update=updates)

    def to(self, device: torch.device) -> "TensorDataModel":
        return self._map_tensors(lambda t: t.to(device))

    def cpu(self) -> "TensorDataModel":
        return self._map_tensors(lambda t: t.cpu())

    def cuda(self) -> "TensorDataModel":
        return self._map_tensors(lambda t: t.cuda())

    def numpy(self) -> "TensorDataModel":
        return self._map_tensors(lambda t: t.detach().cpu().numpy())


class MetadataBase(RepresentationMixin, BaseModel):
    pass


class TensorDataModel(RepresentationMixin, BaseModel):
    """Base model where all declared fields must be tensors."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",  # forbid unknown fields
        validate_assignment=True,
    )

    metadata: Any = Field(default=None, repr=False)
    _is_batched: bool = PrivateAttr(default=False)

    #
    # --------------------------------------------------------------
    # Metadata handling
    # --------------------------------------------------------------
    #
    @classmethod
    def metadata_model(cls) -> type[BaseModel] | None:
        """Return the user-defined inner Metadata class if present."""
        return getattr(cls, "Metadata", None)

    @model_validator(mode="before")
    @classmethod
    def split_metadata(cls, data: Any) -> dict[str, Any]:
        """
        - Pull out keys belonging to Metadata schema
        - Everything else must match declared tensor fields
        """
        if not isinstance(data, dict):
            return data

        meta_cls = cls.metadata_model()
        declared_tensor_fields = set(cls.model_fields.keys()) - {"metadata"}

        metadata_fields = set(meta_cls.model_fields.keys()) if meta_cls else set()

        metadata = {}
        cleaned = {}

        for key, value in data.items():
            if key in declared_tensor_fields:
                cleaned[key] = value
            elif key in metadata_fields:
                metadata[key] = value
            elif key == "metadata":
                # user explicitly passed metadata model
                metadata = value
            else:
                raise ValueError(
                    f"Unexpected field '{key}'. Allowed: {declared_tensor_fields | metadata_fields}"
                )

        if meta_cls:
            cleaned["metadata"] = meta_cls(**metadata)
        else:
            cleaned["metadata"] = None

        return cleaned

    #
    # --------------------------------------------------------------
    # Tensor validation
    # --------------------------------------------------------------
    #
    @model_validator(mode="after")
    def validate_tensor_fields(self) -> Self:
        for name, _ in self.__class__.model_fields.items():
            if name == "metadata":
                continue

            value = getattr(self, name)
            if value is not None and not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"Field '{name}' must be torch.Tensor, got {type(value).__name__}"
                )

        return self

    @classmethod
    def batch(cls, items: list[Self]) -> Self:
        """Create a batched instance from a list of instances."""
        if not items:
            raise ValueError("Cannot batch empty list")

        if not all(type(item) is type(items[0]) for item in items):
            raise TypeError("All items must be of the same type")

        field_values = {}

        for field_name in cls.model_fields.keys():
            if field_name == "metadata":
                # Batch metadata as lists
                batched_meta = {}
                for item in items:
                    for k, v in item.metadata.items():
                        batched_meta.setdefault(k, []).append(v)
                field_values[field_name] = batched_meta
                continue

            vals = [getattr(item, field_name) for item in items]

            if vals[0] is None:
                field_values[field_name] = None
            else:
                field_values[field_name] = torch.stack(vals, dim=0)

        batched_instance = cls(**field_values)
        batched_instance._is_batched = True
        return batched_instance

    def __len__(self):
        """Return batch size if batched, else 1."""
        if not self._is_batched:
            return 1

        for field_name in self.__class__.model_fields.keys():
            if field_name == "metadata":
                continue
            val = getattr(self, field_name)
            if isinstance(val, torch.Tensor):
                return val.shape[0]
        return 1

    def __rich_repr__(self):
        yield from super().__rich_repr__()
        yield "is_batched", self._is_batched
        yield "batch_size", len(self) if self._is_batched else 1


T_TensorDataModel = TypeVar("T_TensorDataModel", bound=TensorDataModel)
