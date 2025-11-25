"""Constants Module

This module defines default constants used throughout the Atria Datasets package,
including default cache directories and download paths.
"""

import os
from pathlib import Path

DEFAULT_ATRIA_CACHE_DIR = os.environ.get(
    "DEFAULT_ATRIA_CACHE_DIR", Path.home() / ".cache/atria/"
)
"""
str: The default directory for caching Atria resources. This value can be overridden
by setting the `ATRIA_CACHE_DIR` environment variable. If not set, it defaults to
`~/.cache/atria/`.
"""

_DEFAULT_ATRIA_DATASETS_CACHE_DIR = Path(DEFAULT_ATRIA_CACHE_DIR) / "datasets/"
"""
str: The default directory for caching Atria resources. This value can be overridden
by setting the `ATRIA_CACHE_DIR` environment variable. If not set, it defaults to
`~/.cache/atria/`.
"""

_DEFAULT_ATRIA_FILE_STORAGE_DIR = Path(DEFAULT_ATRIA_CACHE_DIR) / "fs/"
"""
Path: The default directory for Atria file storage. This is a subdirectory of
`_DEFAULT_ATRIA_CACHE_DIR` and is used to store file system-related resources.
"""

_DEFAULT_DOWNLOAD_PATH = ".download_cache"
"""
Path: The default download path for temporary storage of downloaded files. This
directory is used to store files temporarily during download operations.
"""
