from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.dataset_config import DatasetConfig
    from ..models.model_config import ModelConfig


T = TypeVar("T", bound="InferenceTaskConfig")


@_attrs_define
class InferenceTaskConfig:
    """
    Attributes:
        dataset (DatasetConfig):
        model (ModelConfig):
        experiment_id (None | Unset | UUID):
        sample_ids (list[str] | None | Unset):
    """

    dataset: DatasetConfig
    model: ModelConfig
    experiment_id: None | Unset | UUID = UNSET
    sample_ids: list[str] | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        dataset = self.dataset.to_dict()

        model = self.model.to_dict()

        experiment_id: None | str | Unset
        if isinstance(self.experiment_id, Unset):
            experiment_id = UNSET
        elif isinstance(self.experiment_id, UUID):
            experiment_id = str(self.experiment_id)
        else:
            experiment_id = self.experiment_id

        sample_ids: list[str] | None | Unset
        if isinstance(self.sample_ids, Unset):
            sample_ids = UNSET
        elif isinstance(self.sample_ids, list):
            sample_ids = self.sample_ids

        else:
            sample_ids = self.sample_ids

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "dataset": dataset,
                "model": model,
            }
        )
        if experiment_id is not UNSET:
            field_dict["experiment_id"] = experiment_id
        if sample_ids is not UNSET:
            field_dict["sample_ids"] = sample_ids

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.dataset_config import DatasetConfig
        from ..models.model_config import ModelConfig

        d = dict(src_dict)
        dataset = DatasetConfig.from_dict(d.pop("dataset"))

        model = ModelConfig.from_dict(d.pop("model"))

        def _parse_experiment_id(data: object) -> None | Unset | UUID:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                experiment_id_type_0 = UUID(data)

                return experiment_id_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | Unset | UUID, data)

        experiment_id = _parse_experiment_id(d.pop("experiment_id", UNSET))

        def _parse_sample_ids(data: object) -> list[str] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                sample_ids_type_0 = cast(list[str], data)

                return sample_ids_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[str] | None | Unset, data)

        sample_ids = _parse_sample_ids(d.pop("sample_ids", UNSET))

        inference_task_config = cls(
            dataset=dataset,
            model=model,
            experiment_id=experiment_id,
            sample_ids=sample_ids,
        )

        inference_task_config.additional_properties = d
        return inference_task_config

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
