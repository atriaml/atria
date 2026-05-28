import fire
from atria_datasets.api.datasets import FileStorageType, load_dataset_config


def main(dataset_name: str = "tobacco3482/image_with_ocr"):
    # load example dataset
    dataset_config = load_dataset_config(dataset_name)

    # build the dataset and model pipeline
    dataset = dataset_config.build(
        data_dir="/mnt/noel/phd-2026/.atria_cache/",
        cached_storage_type=FileStorageType.DELTALAKE,
        enable_cached_splits=True,
    )

    print("dataset", dataset)


if __name__ == "__main__":
    fire.Fire(main)
