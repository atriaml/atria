from atria_models.core.model_pipelines._image_pipeline import *  # noqa
from atria_models.core.model_pipelines._model_pipeline import *  # noqa
from atria_models.core.model_pipelines._sequence_pipeline import *  # noqa
from atria_models.core.models.transformers._models._bert import *  # noqa
from atria_models import MODEL_PIPELINES, MODELS

if __name__ == "__main__":
    MODEL_PIPELINES.dump(refresh=True)
    MODELS.dump()
