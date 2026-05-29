from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

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
        is_metrics_computation_run (Union[Unset, bool]):  Default: False.
    """

    dataset: "DatasetConfig"
    model: "ModelConfig"
    is_metrics_computation_run: Union[Unset, bool] = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        dataset = self.dataset.to_dict()

        model = self.model.to_dict()

        is_metrics_computation_run = self.is_metrics_computation_run

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

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.dataset_config import DatasetConfig
        from ..models.model_config import ModelConfig

        d = dict(src_dict)
        dataset = DatasetConfig.from_dict(d.pop("dataset"))

        model = ModelConfig.from_dict(d.pop("model"))

        is_metrics_computation_run = d.pop("is_metrics_computation_run", UNSET)

        evaluation_task_config = cls(
            dataset=dataset,
            model=model,
            is_metrics_computation_run=is_metrics_computation_run,
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
