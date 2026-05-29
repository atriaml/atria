from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.config_base import ConfigBase
    from ..models.dataset_config import DatasetConfig
    from ..models.model_config import ModelConfig


T = TypeVar("T", bound="ExplanationTaskConfig")


@_attrs_define
class ExplanationTaskConfig:
    """
    Attributes:
        dataset (DatasetConfig):
        model (ModelConfig):
        explainer_pipeline_config (ConfigBase):
    """

    dataset: "DatasetConfig"
    model: "ModelConfig"
    explainer_pipeline_config: "ConfigBase"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        dataset = self.dataset.to_dict()

        model = self.model.to_dict()

        explainer_pipeline_config = self.explainer_pipeline_config.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "dataset": dataset,
                "model": model,
                "explainer_pipeline_config": explainer_pipeline_config,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.config_base import ConfigBase
        from ..models.dataset_config import DatasetConfig
        from ..models.model_config import ModelConfig

        d = dict(src_dict)
        dataset = DatasetConfig.from_dict(d.pop("dataset"))

        model = ModelConfig.from_dict(d.pop("model"))

        explainer_pipeline_config = ConfigBase.from_dict(d.pop("explainer_pipeline_config"))

        explanation_task_config = cls(
            dataset=dataset,
            model=model,
            explainer_pipeline_config=explainer_pipeline_config,
        )

        explanation_task_config.additional_properties = d
        return explanation_task_config

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
