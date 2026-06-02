from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.dataset_config import DatasetConfig
    from ..models.model_config import ModelConfig


T = TypeVar("T", bound="EvaluationTaskConfig")


@_attrs_define
class EvaluationTaskConfig:
    """
    Attributes:
        dataset (DatasetConfig):
        model (ModelConfig):
        is_metrics_computation_run (bool | Unset):  Default: False.
        experiment_name (None | str | Unset):
        run_name (None | str | Unset):
    """

    dataset: DatasetConfig
    model: ModelConfig
    is_metrics_computation_run: bool | Unset = False
    experiment_name: None | str | Unset = UNSET
    run_name: None | str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        dataset = self.dataset.to_dict()

        model = self.model.to_dict()

        is_metrics_computation_run = self.is_metrics_computation_run

        experiment_name: None | str | Unset
        if isinstance(self.experiment_name, Unset):
            experiment_name = UNSET
        else:
            experiment_name = self.experiment_name

        run_name: None | str | Unset
        if isinstance(self.run_name, Unset):
            run_name = UNSET
        else:
            run_name = self.run_name

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "dataset": dataset,
                "model": model,
            }
        )
        if is_metrics_computation_run is not UNSET:
            field_dict["is_metrics_computation_run"] = is_metrics_computation_run
        if experiment_name is not UNSET:
            field_dict["experiment_name"] = experiment_name
        if run_name is not UNSET:
            field_dict["run_name"] = run_name

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.dataset_config import DatasetConfig
        from ..models.model_config import ModelConfig

        d = dict(src_dict)
        dataset = DatasetConfig.from_dict(d.pop("dataset"))

        model = ModelConfig.from_dict(d.pop("model"))

        is_metrics_computation_run = d.pop("is_metrics_computation_run", UNSET)

        def _parse_experiment_name(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        experiment_name = _parse_experiment_name(d.pop("experiment_name", UNSET))

        def _parse_run_name(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        run_name = _parse_run_name(d.pop("run_name", UNSET))

        evaluation_task_config = cls(
            dataset=dataset,
            model=model,
            is_metrics_computation_run=is_metrics_computation_run,
            experiment_name=experiment_name,
            run_name=run_name,
        )

        evaluation_task_config.additional_properties = d
        return evaluation_task_config

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
