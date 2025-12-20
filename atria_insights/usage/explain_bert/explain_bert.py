from atria_datasets.api.datasets import load_dataset_config  # noqa: F401
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa: F401
from atria_logger import get_logger
from atria_models import MODEL_PIPELINES  # noqa: F401
from atria_models.api.models import load_model_pipeline_config  # noqa: F401
from atria_models.core.model_builders._common import ModelBuilderType  # noqa: F401
from atria_models.core.model_pipelines._common import ModelConfig  # noqa: F401
from atria_transforms.api.tfs import load_transform  # noqa: F401

from atria_insights.core.configs.explainer_config import (
    DataConfig,  # noqa: F401
    ExplainerRunConfig,  # noqa: F401
    RuntimeEnvConfig,  # noqa: F401
)
from atria_insights.core.model_explainer import ModelExplainer
from atria_insights.core.model_pipelines._api import (
    load_x_model_pipeline_config,  # noqa: F401
)

logger = get_logger(__name__)

print(MODEL_PIPELINES.list_all_modules())

# Example usage: Explain a BERT model on a text classification task
task_type = "sequence_classification"

# create env config
env_config = RuntimeEnvConfig(
    project_name="explain_bert", run_name="run_001", output_dir="./outputs/", seed=42
)

# Load data transforms
transforms = load_transform(
    f"{task_type}_document_processor",
    hf_processor={"tokenizer_name": "bert-base-uncased"},
    image_transform={"stats": "imagenet"},
)

# load model and model pipeline configs
model_pipeline_config = load_model_pipeline_config(
    task_type,
    model=ModelConfig(
        model_name_or_path="bert-base-uncased",
        builder_type=ModelBuilderType.atria,
        model_type=task_type,
    ),
    train_transform=transforms,
    eval_transform=transforms,
)

# load data config and run config
data_config = DataConfig(
    dataset_config=load_dataset_config("tobacco3482/image_with_ocr"),
    num_workers=0,
    train_batch_size=4,
    eval_batch_size=4,
)

config = ExplainerRunConfig(
    env_config=env_config,
    x_model_pipeline_config=load_x_model_pipeline_config(
        task_type, model_pipeline_config=model_pipeline_config
    ),
    data_config=data_config,
)
model_explainer = ModelExplainer(config=config)
model_explainer.run()
