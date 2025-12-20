"""Constants"""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_ATRIA_TFS_CACHE_DIR = os.environ.get(
    "DEFAULT_ATRIA_CACHE_DIR", Path.home() / ".cache/atria/tfs/"
)
"""
str: The default directory for caching Atria resources. This value can be overridden
by setting the `ATRIA_CACHE_DIR` environment variable. If not set, it defaults to
`~/.cache/atria/tfs/`.
"""
