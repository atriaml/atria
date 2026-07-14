from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_features(n: int, center: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = rng.normal(loc=center, scale=0.3, size=(n, 1))
    return pd.DataFrame(values, columns=["feature_0"])


@pytest.fixture
def separable_features() -> dict[str, pd.DataFrame]:
    """4 feature splits with a clearly-separable single feature (members ~1.0, non-members ~0.0)."""
    return {
        "features_members_train": _make_features(200, center=1.0, seed=1),
        "features_nonmembers_train": _make_features(200, center=0.0, seed=2),
        "features_members_test": _make_features(100, center=1.0, seed=3),
        "features_nonmembers_test": _make_features(100, center=0.0, seed=4),
    }
