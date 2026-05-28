from typing import Literal

import fire
from atria_datasets.api.datasets import load_dataset_config
from atria_models.api.models import load_model_pipeline_config
from atria_models.core.model_builders._common import ModelBuilderType
from atria_models.core.model_pipelines._common import ModelConfig
from atria_transforms.api.tfs import load_transform
from atria_transforms.tfs._image_transforms import StandardImageTransform


def main(
    dataset_name: str = "tobacco3482/image_with_ocr",
    model_name: str = "bert-base-uncased",
    tokenizer_name: str = "bert-base-uncased",
    builder_type: ModelBuilderType = ModelBuilderType.atria,
    stats: Literal["imagenet", "standard", "openai_clip", "custom"] = "standard",
    image_size: int = 224,
):
    # load example model
    model_pipeline_config = load_model_pipeline_config(
        "sequence_classification",
        model=ModelConfig(
            model_name_or_path=model_name,
            builder_type=builder_type,
            model_type="sequence_classification",
        ),
        train_transform=load_transform(
            "document_processor/sequence_classification",
            hf_processor={"tokenizer_name": tokenizer_name},
            image_transform=StandardImageTransform(
                stats=stats, resize_width=image_size, resize_height=image_size
            ),
            overflow_strategy="return_first",
        ),
        eval_transform=load_transform(
            "document_processor/sequence_classification",
            hf_processor={"tokenizer_name": tokenizer_name},
            image_transform=StandardImageTransform(
                stats=stats, resize_width=image_size, resize_height=image_size
            ),
            overflow_strategy="return_first",
        ),
    )

    # load example dataset
    dataset_config = load_dataset_config(dataset_name)

    # build the dataset and model pipeline
    dataset = dataset_config.build()

    # example labels for building the model pipeline
    model = model_pipeline_config.build()

    print("model_pipeline", model_pipeline)


if __name__ == "__main__":
    fire.Fire(main)
