# from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa: F401
# from atria_registry._module_registry import ModuleRegistry

# for group in ModuleRegistry().registry_groups.values():
#     print(f"Registry Group: {group.name}")
#     for module_name in group.list_registered_modules():
#         print(f"  Module: {module_name}")

# # class ModelPipelineConfig(RepresentationMixin, BaseModel):
# #     model_pipeline: str
# #     model_pipeline_kwargs: dict[str, object] = {}

# #     def model_post_init(self, context: Any) -> None:
# #         self._model_pipeline_config = load_model_pipeline_config(
# #             self.model_pipeline, **self.model_pipeline_kwargs
# #         )

# #     def build_model_pipeline(self, labels: DatasetLabels) -> ModelPipeline:
# #         return self._model_pipeline_config.build(labels=labels)


# # class TrainerConfig(BaseModel):
# #     model_pipeline: ModelPipelineConfig


# # # config = load_model_pipeline_config(
# # #     "image_classification",
# # #     model=ModelConfig(
# # #         model_name_or_path="resnet18",
# # #     ),
# # # )

# # # print("config", config)
# # # print(config.to_yaml())
# # config = TrainerConfig(
# #     model_pipeline=ModelPipelineConfig(
# #         model_pipeline="image_classification",
# #         model_pipeline_kwargs={
# #             "model": {
# #                 "model_name_or_path": "resnet18",
# #             }
# #         },
# #     )
# # )

# # config = TrainerConfig(
# #     model_pipeline=load_model_pipeline_config(
# #         "image_classification",
# #         model=ModelConfig(
# #             model_name_or_path="resnet18",
# #         ),
# #     )
# # )
# # print(config)
# # print(
# #     "config",
# #     print(config.model._model_pipeline_config),
# #     # config.model.build_model_pipeline(
# #     #     labels=DatasetLabels(classification=["cat", "dog"])
# #     # ),
# # )
