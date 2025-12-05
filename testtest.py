from atria_datasets.api.datasets import load_dataset_config
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa: F401
from atria_ml.task_pipelines.configs._trainer import DataConfig
from atria_models.api.models import load_model_pipeline_config
from atria_models.core.model_pipelines._common import ModelConfig, ModelPipelineConfig
from pydantic import BaseModel

# class ModelPipelineConfig(RepresentationMixin, BaseModel):
#     model_pipeline: str
#     model_pipeline_kwargs: dict[str, object] = {}

#     def model_post_init(self, context: Any) -> None:
#         self._model_pipeline_config = load_model_pipeline_config(
#             self.model_pipeline, **self.model_pipeline_kwargs
#         )

#     def build_model_pipeline(self, labels: DatasetLabels) -> ModelPipeline:
#         return self._model_pipeline_config.build(labels=labels)


class TrainerConfig(BaseModel):
    model_pipeline: ModelPipelineConfig
    data: DataConfig


config = TrainerConfig(
    model_pipeline=load_model_pipeline_config(
        "image_classification",
        model=ModelConfig(
            model_name_or_path="resnet18",
        ),
        # train_transform=load_transform("image_processor"),
        # eval_transform=load_transform("image_processor"),
    ),
    data=DataConfig(
        dataset_config=load_dataset_config(
            "cifar10/1k",
        )
    ),
)


print(config)
