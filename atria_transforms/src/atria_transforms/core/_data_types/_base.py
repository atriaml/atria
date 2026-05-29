"""Base types for tensor data models."""

from typing import Any, Self, TypeVar

import numpy as np
from atria_types._utilities._repr import RepresentationMixin
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from atria_transforms.core._data_types._ops import TensorOperations


class TensorDataModel(RepresentationMixin, BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, extra="forbid", validate_assignment=True
    )

    metadata: Any = Field(default=None, repr=False)
    _is_batched: bool = PrivateAttr(default=False)

    @property
    def ops(self) -> TensorOperations:
        return TensorOperations(self)

    @classmethod
    def metadata_model(cls) -> type[BaseModel] | None:
        Metadata = getattr(cls, "Metadata", None)
        assert Metadata is None or issubclass(Metadata, BaseModel)
        return Metadata

    @model_validator(mode="before")
    @classmethod
    def split_metadata(cls, data: Any) -> dict[str, Any]:
        if not isinstance(data, dict):
            return data

        is_batched = data.pop("is_batched", False)
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
                metadata = value
            else:
                raise ValueError(
                    f"Unexpected field '{key}'. Allowed: {declared_tensor_fields | metadata_fields}"
                )

        if meta_cls:
            cleaned["metadata"] = (
                meta_cls.model_construct(**metadata)
                if is_batched
                else meta_cls(**metadata)
            )
        else:
            cleaned["metadata"] = None

        if is_batched:
            batch_sizes = set()
            for name, value in cleaned.items():
                if name == "metadata":
                    continue
                if value is not None and hasattr(value, "shape"):
                    batch_sizes.add(value.shape[0])

            if len(batch_sizes) > 1:
                raise ValueError(
                    f"All fields must have the same batch size. Found: {batch_sizes}"
                )
        return cleaned

    @model_validator(mode="after")
    def validate_tensor_fields(self) -> Self:
        import torch

        for name in self.__class__.model_fields:
            if name == "metadata":
                continue
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, (np.ndarray, torch.Tensor)):
                    raise TypeError(
                        f"Field '{name}' must be np.ndarray or torch.Tensor, "
                        f"got {type(value).__name__}"
                    )
        return self

    @classmethod
    def batch(cls, items: list[Self]) -> Self:
        if not items:
            raise ValueError("Cannot batch empty list")

        if not all(type(item) is type(items[0]) for item in items):
            raise TypeError("All items must be of the same type")

        field_values = {}

        for field_name in cls.model_fields.keys():
            if field_name == "metadata":
                batched_meta = {}
                for item in items:
                    for k, v in item.metadata.model_dump().items():
                        batched_meta.setdefault(k, []).append(v)
                field_values[field_name] = batched_meta
                continue

            vals = [getattr(item, field_name) for item in items]

            if vals[0] is None:
                field_values[field_name] = None
            elif isinstance(vals[0], np.ndarray):
                field_values[field_name] = np.stack(vals, axis=0)
            else:
                import torch

                field_values[field_name] = torch.stack(vals, dim=0)

        batched_instance = cls(**field_values, is_batched=True)
        batched_instance._is_batched = True
        return batched_instance

    def __len__(self):
        if not self._is_batched:
            return 1

        for field_name in self.__class__.model_fields.keys():
            if field_name == "metadata":
                continue
            val = getattr(self, field_name)
            if val is not None and hasattr(val, "shape"):
                return val.shape[0]
        return 1

    def __rich_repr__(self):
        yield from super().__rich_repr__()
        yield "is_batched", self._is_batched
        yield "batch_size", len(self) if self._is_batched else 1


T_TensorDataModel = TypeVar("T_TensorDataModel", bound=TensorDataModel | Any)
