import pickle
from pathlib import Path
from typing import Any


class PickleCacher:
    def __init__(self, cache_dir: str) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def exists(self, sample_id: str) -> bool:
        cache_file = self._cache_dir / f"{sample_id}.pkl"
        return cache_file.exists()

    def load(self, sample_id: str) -> Any | None:
        cache_file = self._cache_dir / f"{sample_id}.pkl"

        with open(cache_file, "rb") as f:
            return pickle.load(f)

    def save(self, sample_id: str, data: Any) -> None:
        cache_file = self._cache_dir / f"{sample_id}.pkl"
        import pickle

        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
