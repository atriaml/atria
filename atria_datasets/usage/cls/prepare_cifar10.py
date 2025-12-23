# import os

# from atria_logger import get_logger

# from atria_datasets import load_dataset_config

# logger = get_logger(__name__)


# SCRIPT_PATH = os.path.abspath(__file__)
# SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)


# def main():
#     dataset_config = load_dataset_config("cifar10/default")
#     dataset = dataset_config.build(enable_cached_splits=True)
#     logger.info(f"Loaded dataset:\n{dataset}")

#     # get first sample
#     sample = next(iter(dataset.train))

#     logger.info(f"First sample in train split:\n{sample}")

#     sample.viz.visualize(
#         output_path=os.path.join(SCRIPT_DIR, "visualizations/{}").format(
#             dataset.config.dataset_name
#         )
#     )


# if __name__ == "__main__":
#     main()

from atria_datasets.registry.image_classification.cifar10 import Cifar10

dataset = Cifar10()
print(dataset)
