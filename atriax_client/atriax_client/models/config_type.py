from enum import Enum


class ConfigType(str, Enum):
    BATCH_SAMPLER = "batch_sampler"
    DATASET = "dataset"
    DATASET_SPLITTER = "dataset_splitter"
    DATASET_STORAGE_MANAGER = "dataset_storage_manager"
    DATA_PIPELINE = "data_pipeline"
    DATA_TRANSFORM = "data_transform"
    ENGINE = "engine"
    ENGINE_STEP = "engine_step"
    EXPLAINER = "explainer"
    EXPLAINER_METRIC = "explainer_metric"
    EXPLAINER_PIPELINE = "explainer_pipeline"
    FEATURE_PERTURBORS = "feature_perturbors"
    IMAGE_SEGMENTOR = "image_segmentor"
    LR_SCHEDULER_FACTORY = "lr_scheduler_factory"
    METRIC_FACTORY = "metric_factory"
    MODEL = "model"
    MODEL_PIPELINE = "model_pipeline"
    OPTIMIZER_FACTORY = "optimizer_factory"
    TASK_PIPELINE = "task_pipeline"

    def __str__(self) -> str:
        return str(self.value)
