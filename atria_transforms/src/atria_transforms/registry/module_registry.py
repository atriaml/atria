# from atria_registry.module_registry import ModuleRegistry

# from atria_transforms.registry.registry_groups import DataTransformRegistryGroup

# _initialized = False


# def init_registry():
#     global _initialized
#     if _initialized:
#         return
#     _initialized = True
#     ModuleRegistry().add_registry_group(
#         name="DATA_TRANSFORM",
#         registry_group=DataTransformRegistryGroup(
#             name="data_transform", default_provider="atria_transforms"
#         ),
#     )
