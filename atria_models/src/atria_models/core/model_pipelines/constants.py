from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ATRIA_MODELS_CACHE_DIR = (
    Path(os.environ.get("DEFAULT_ATRIA_CACHE_DIR", Path.home() / ".cache/atria/")) / "models/"
)
_DEFAULT_ATRIA_MODELS_STORAGE_SUBDIR = "storage"
_DEFAULT_MODEL_METADATA_PATH = "metadata.yaml"
_DEFAULT_MODEL_WEIGHTS_PATH = "model.safetensors"
