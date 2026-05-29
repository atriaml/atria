import os

from atria_datasets.core.storage.utilities import FileStorageType
import fire
from atria_logger import get_logger
from atria_transforms import load_transform

from atria_datasets import load_dataset_config

logger = get_logger(__name__)

SCRIPT_PATH = os.path.abspath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)

TRANSFORM_KWARGS = dict(stats="imagenet", resize_height=32, resize_width=32)


def main(case: int = 0):
    dataset_config = load_dataset_config("cifar10/default")

    train_tf = load_transform("image_processor", tf=TRANSFORM_KWARGS)
    eval_tf = load_transform("image_processor", tf=TRANSFORM_KWARGS)

    if case == 0:
        # Raw in-memory load — returns Dataset
        logger.info("=== Case 0: raw in-memory load ===")
        dataset = dataset_config.build(enable_cached_splits=False)
        logger.info(f"Dataset: {dataset}")
        sample = next(iter(dataset.train))
        logger.info(f"First train sample:\n{sample}")

    elif case == 1:
        # Cache after raw build — explicit .cache() on a loaded Dataset
        logger.info("=== Case 2: cache after raw load ===")
        dataset = dataset_config.build(enable_cached_splits=False)
        logger.info(f"Dataset: {dataset}")
        cached = dataset.cache()
        logger.info(f"CachedDataset: {cached}")
        sample = next(iter(cached.train))
        logger.info(f"First train sample from post-build cache:\n{sample}")

    elif case == 2:
        # Cached load — returns CachedDataset (cache-miss writes; re-run hits fast path)
        logger.info("=== Case 1: load with caching ===")
        cached = dataset_config.build(enable_cached_splits=True)
        logger.info(f"CachedDataset: {cached}")
        sample = next(iter(cached.train))
        logger.info(f"First train sample from cache:\n{sample}")

    elif case == 3:
        # Load with caching + transform — transform hash baked into cache path
        logger.info("=== Case 3: load with caching + transform ===")
        cached = dataset_config.build(
            enable_cached_splits=True,
            train_transform=train_tf,
            eval_transform=eval_tf,
        )
        logger.info(f"CachedDataset: {cached}")
        sample = next(iter(cached.train))
        logger.info(f"First train sample from transform cache:\n{sample}")

    elif case == 4:
        # Load with caching + transform — transform hash baked into cache path
        logger.info("=== Case 4: load with caching + transform + MSGPACK ===")
        cached = dataset_config.build(
            enable_cached_splits=True,
            train_transform=train_tf,
            eval_transform=eval_tf,
            cached_storage_type=FileStorageType.MSGPACK,
        )
        logger.info(f"CachedDataset: {cached}")
        sample = next(iter(cached.train))
        logger.info(f"First train sample from transform cache:\n{sample}")

    elif case == 5:
        # Raw in-memory load + transform — returns Dataset, transforms applied on-the-fly
        logger.info("=== Case 5: raw in-memory load + transform ===")
        dataset = dataset_config.build(enable_cached_splits=False, train_transform=train_tf, eval_transform=eval_tf)
        logger.info(f"Dataset: {dataset}")
        sample = next(iter(dataset.train))
        logger.info(f"First train sample:\n{sample}")

    elif case == 6:
        # CachedDataset.process() — re-cache existing CachedDataset with transforms
        logger.info("=== Case 4: CachedDataset.process() ===")
        cached = dataset_config.build(enable_cached_splits=True)
        logger.info(f"CachedDataset: {cached}")
        sample = next(iter(cached.train))
        logger.info(f"First train sample from processed cache:\n{sample}")

    elif case == 7:
        # CachedDataset.process() — re-cache existing CachedDataset with transforms
        logger.info("=== Case 4: CachedDataset.process() ===")
        cached = dataset_config.build(enable_cached_splits=True)
        logger.info(f"CachedDataset: {cached}")
        processed = cached.process(train_transform=train_tf, eval_transform=eval_tf, cached_storage_type=FileStorageType.MSGPACK)
        logger.info(f"Processed CachedDataset: {processed}")
        sample = next(iter(cached.train))
        logger.info(f"First train sample from processed cache:\n{sample}")
    else:
        raise ValueError(f"Unknown case {case}. Valid cases: 0–4.")


if __name__ == "__main__":
    fire.Fire(main)
