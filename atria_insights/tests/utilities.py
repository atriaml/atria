from typing import Any

import numpy as np
import torch


def _assert_data_equal(data1: dict[str, Any], data2: dict[str, Any]) -> None:
    """Assert two data dictionaries are equal, handling different types."""
    assert data1.keys() == data2.keys()

    for key in data1.keys():
        val1, val2 = data1[key], data2[key]

        if isinstance(val1, torch.Tensor):
            assert torch.allclose(val1, val2), (
                f"Tensors not equal for key '{key}', Failed values: {val1} vs {val2}"
            )
        elif isinstance(val1, np.ndarray):
            assert np.allclose(val1, val2), (
                f"Arrays not equal for key '{key}', Failed values: {val1} vs {val2}"
            )
        elif isinstance(val1, (list, tuple)):
            assert len(val1) == len(val2), (
                f"Length mismatch for key '{key}', Found lengths: {len(val1)} vs {len(val2)}"
            )
            for i, (item1, item2) in enumerate(zip(val1, val2, strict=True)):
                if isinstance(item1, torch.Tensor):
                    assert torch.allclose(item1, item2), (
                        f"Tensor mismatch at {key}[{i}], Values: {item1} vs {item2}"
                    )
                elif isinstance(item1, np.ndarray):
                    assert np.allclose(item1, item2), (
                        f"Array mismatch at {key}[{i}], Values: {item1} vs {item2}"
                    )
                else:
                    assert item1 == item2, (
                        f"Value mismatch at {key}[{i}], Values: {item1} vs {item2}"
                    )
        elif isinstance(val1, dict):
            _assert_data_equal(val1, val2)  # Recursive for nested dicts
        else:
            assert val1 == val2, f"Values not equal for key '{key}': {val1} != {val2}"
