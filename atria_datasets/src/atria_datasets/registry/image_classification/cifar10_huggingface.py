from typing import Any

from atria_types import ClassificationAnnotation, Image, ImageInstance, Label

from atria_datasets import DATASET, HuggingfaceDatasetConfig, HuggingfaceImageDataset


@DATASET.register(
    "huggingface_cifar10",
    configs=[
        HuggingfaceDatasetConfig(
            config_name="plain_text",
            hf_repo="uoft-cs/cifar10",
            hf_config_name="plain_text",
        ),
        HuggingfaceDatasetConfig(
            config_name="plain_text_1k",
            hf_repo="uoft-cs/cifar10",
            hf_config_name="plain_text",
            max_train_samples=1000,
            max_test_samples=1000,
            max_validation_samples=1000,
        ),
    ],
)
class HuggingfaceCifar10(HuggingfaceImageDataset):
    def _input_transform(self, sample: dict[str, Any]) -> ImageInstance:
        return ImageInstance(
            image=Image(content=sample["img"]),
            annotations=[
                ClassificationAnnotation(
                    label=Label(
                        value=sample["label"],
                        name=self.metadata.dataset_labels.classification[
                            sample["label"]
                        ],
                    )
                )
            ],
        )
