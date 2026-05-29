import fire
from atria_datasets.api.datasets import FileStorageType, load_dataset_config


def main(dataset_name: str = "cifar10/1k"):
    # load example dataset
    dataset_config = load_dataset_config(dataset_name)

    # build the dataset and model pipeline
    dataset = dataset_config.build(
        cached_storage_type=FileStorageType.DELTALAKE,
        enable_cached_splits=True,
    )

    print("dataset", dataset)

    dataset.cache(
        train_transform=dataset_config.model_config.train_transform,
        eval_transform=dataset_config.model_config.eval_transform,
    )


if __name__ == "__main__":
    fire.Fire(main)
