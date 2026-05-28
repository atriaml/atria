import os

from atria_logger import get_logger

from atria_datasets import load_dataset_config
from atria_datasets.core.dataset._cached_dataset import CachedDataset

logger = get_logger(__name__)


SCRIPT_PATH = os.path.abspath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)


def main(visualize_sample: bool = False, upload: bool = True):
    dataset_config = load_dataset_config("cifar10/default")
    dataset = dataset_config.build(enable_cached_splits=True)
    logger.info(f"Loaded dataset:\n{dataset}")

    # get first sample
    sample = next(iter(dataset.train))

    logger.info(f"First sample in train split:\n{sample}")

    if visualize_sample:
        sample.viz.visualize(
            output_path=os.path.join(SCRIPT_DIR, "visualizations/{}").format(
                dataset.config.dataset_name
            )
        )

    if upload:
        repo_info = dataset.upload_to_hub(name="cifar10-example2")
        print("repo_info", repo_info)

        # # reload dataset from hub to verify upload
        dataset = CachedDataset.load_from_hub(
            username=repo_info["username"],
            name=repo_info["name"],
            branch=repo_info["branch"],
        )

        logger.info(f"dataset loaded from hub successfully: \n{dataset}")
        # hub_sample = next(iter(hub_dataset.train))
        # logger.info(f"First sample in train split of hub dataset:\n{hub_sample}")


if __name__ == "__main__":
    main()
