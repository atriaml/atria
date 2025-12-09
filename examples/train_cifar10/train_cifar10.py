from atria_datasets.api.datasets import load_dataset_config
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa: F401
from atria_ml.task_pipelines._trainer import Trainer
from atria_ml.task_pipelines.configs._base import (
    DataConfig,
    RunConfig,
    RuntimeEnvConfig,
    TrainerConfig,
)
from atria_models.api.models import load_model_pipeline_config
from atria_models.core.model_pipelines._common import ModelConfig
from atria_transforms.api.tfs import load_transform

config = RunConfig(
    env=RuntimeEnvConfig(
        project_name="my_atria_project",
        run_name="cifar10_experiment_03",
        output_dir="./outputs/",
        seed=42,
    ),
    model_pipeline=load_model_pipeline_config(
        "image_classification",
        model=ModelConfig(model_name_or_path="resnet50"),
        train_transform=load_transform(
            "image_processor",
            tf={
                "use_imagenet_mean_std": True,
            },
        ),
        eval_transform=load_transform(
            "image_processor", tf={"use_imagenet_mean_std": True}
        ),
    ),
    data=DataConfig(
        dataset_config=load_dataset_config("cifar10/default"),
        data_dir="data_dir/",
        num_workers=0,
        train_batch_size=64,
        eval_batch_size=64,
    ),
    trainer=TrainerConfig(),
    do_train=True,
)
# exit()
# config_dict_before = config.model_dump()
# config.save_to_json()
# config.from_json(Path(config.env.output_dir) / "config.json")
# config_dict_after = config.model_dump()
# if config_dict_before != config_dict_after:
#     raise ValueError("Config serialization/deserialization failed!")
# else:
#     print("Config serialization/deserialization successful!")

trainer = Trainer(config=config)
trainer.run()
