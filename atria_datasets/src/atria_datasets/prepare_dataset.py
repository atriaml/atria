import fire
from atria_logger import get_logger

logger = get_logger(__name__)

from atria_datasets.registry.document_classification.rvlcdip import *  # noqa
from atria_datasets.registry.document_classification.tobacco3482 import *  # noqa
from atria_datasets.registry.ser.cord import *  # noqa

# from atria_datasets.registry.ser.docbank import *  # noqa
# from atria_datasets.registry.ser.docile import *  # noqa
from atria_datasets.registry.ser.funsd import *  # noqa

from atria_datasets.registry.ser.sroie import *  # noqa
from atria_datasets.registry.ser.wild_receipts import *  # noqa
# from atria_datasets.registry.vqa.docvqa import *  # noqa
# from atria_datasets.registry.layout_analysis.doclaynet import *  # noqa
# from atria_datasets.registry.layout_analysis.icdar2019 import *  # noqa
# from atria_datasets.registry.layout_analysis.publaynet import *  # noqa
# from atria_datasets.registry.table_extraction.fintabnet import *  # noqa
# from atria_datasets.registry.table_extraction.icdar2013 import *  # noqa
# from atria_datasets.registry.table_extraction.pubtables1m import *  # noqa
# from atria_datasets.registry.vqa.due import *  # noqa


def prepare_dataset(
    name: str,
    data_dir: str | None = None,
    max_samples: int | None = None,
    access_token: str | None = None,
    overwrite_existing_cached: bool = True,
    num_processes: int = 0,
    visualize_samples: bool = True,
    visualized_split: str = "train",
    n_visualized_samples: int = 16,
    print_samples: bool = True,
):
    # first lets load the image classification datasets
    from atria_datasets import load_dataset_config

    dataset_config = load_dataset_config(
        name,
        max_train_samples=max_samples,
        max_test_samples=max_samples,
        max_validation_samples=max_samples,
    )
    logger.info(f"Loaded dataset config:\n{dataset_config}")
    if data_dir is not None:
        data_dir += "/" + name.split("/")[0]
    dataset = dataset_config.build(
        data_dir=data_dir,
        access_token=access_token,
        overwrite_existing_cached=overwrite_existing_cached,
        num_processes=num_processes,
        enable_cached_splits=True,
        max_cache_image_size=1024,
    )
    logger.info(f"Loaded dataset:\n{dataset}")

    if print_samples:
        logger.info(f"Printing one sample from {name} dataset")
        for sample in dataset.train:
            logger.info(sample)
            sample.viz.visualize(
                output_path=f"visualizations/{dataset.config.dataset_name}/train"
            )
            break

    if visualize_samples:
        output_path = f"visualizations/{dataset.config.dataset_name}/train"
        logger.info(
            f"Visualizing {n_visualized_samples} samples from {visualized_split} split to {output_path}"
        )
        for idx, sample in enumerate(dataset._split_iterators[visualized_split]):
            sample.viz.visualize(output_path=output_path)
            if idx + 1 >= n_visualized_samples:
                break


def main():
    fire.Fire(prepare_dataset)


if __name__ == "__main__":
    main()
