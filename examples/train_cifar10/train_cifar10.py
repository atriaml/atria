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

config = RunConfig(
    env=RuntimeEnvConfig(
        project_name="my_atria_project",
        run_name="cifar10_experiment_01",
        output_dir="./outputs/",
        seed=42,
    ),
    model_pipeline=load_model_pipeline_config(
        "image_classification",
        model=ModelConfig(model_name_or_path="resnet18"),
        # train_transform=load_transform("image_processor"),
        # eval_transform=load_transform("image_processor"),
    ),
    data=DataConfig(
        dataset_config=load_dataset_config("cifar10/1k"),
        data_dir="data_dir/",
        num_workers=0,
    ),
    trainer=TrainerConfig(
        max_epochs=100,
    ),
    do_train=False,
)
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
