from atria_models.core.model_pipelines._image_pipeline import *  # noqa
from atria_models.core.model_pipelines._model_pipeline import *  # noqa
from atria_models.core.model_pipelines._sequence_pipeline import *  # noqa
from atria_models import MODEL_PIPELINE

if __name__ == "__main__":
    MODEL_PIPELINE.dump()
