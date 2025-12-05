"""Constants"""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_ATRIA_CACHE_DIR = os.environ.get(
    "DEFAULT_ATRIA_CACHE_DIR", Path.home() / ".cache/atria/"
)
"""
str: The default directory for caching Atria resources. This value can be overridden
by setting the `ATRIA_CACHE_DIR` environment variable. If not set, it defaults to
`~/.cache/atria/`.
"""

_DEFAULT_ATRIA_MODELS_CACHE_DIR = Path(_DEFAULT_ATRIA_CACHE_DIR) / "models/"
"""
str: The default directory for caching Atria models. This value can be overridden
by setting the `ATRIA_CACHE_DIR` environment variable. If not set, it defaults to
`~/.cache/atria/models/`.
"""
