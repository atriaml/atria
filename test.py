from atria_datasets.api.datasets import load_dataset
from atria_datasets.core.dataset._dataset_processor import DatasetProcessor
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa

# dataset = Cifar10()
# print(f"Dataset instantiated: {dataset}")

cifar10_from_registry = load_dataset("cifar10/1k")
print("cifar10_from_registry", cifar10_from_registry)

# now lets say we try to rebuild cache
# cifar10_from_registry
# for sample in cifar10_from_registry.train:
#     print(sample)


post_process_data_dir = "test_data_dir/"
processed = DatasetProcessor(
    dataset=cifar10_from_registry,
    transform=lambda x: x.update(image=x.image.ops.resize(100, 10)),
    split="train",
    processed_data_dir=post_process_data_dir,
).process_splits()

for sample in processed["train"]:
    print(sample)
