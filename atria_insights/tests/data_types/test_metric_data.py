from typing import Any

import pytest
import torch
from pydantic import ValidationError

from atria_insights.data_types._metric_data import BatchMetricData, SampleMetricData
from tests.utilities import _assert_data_equal


@pytest.fixture
def sample_ids() -> list[str]:
    """Sample IDs for testing."""
    return ["sample_001", "sample_002", "sample_003"]


@pytest.mark.parametrize(
    "test_data",
    [
        # Test case 1: Simple scalars
        {
            "accuracy": [0.95, 0.87, 0.92],
            "loss": [0.1, 0.3, 0.2],
            "prediction": [1, 0, 1],
        },
        # Test case 1: Simple scalar tensors
        {
            "accuracy": [torch.tensor(0.95), torch.tensor(0.87), torch.tensor(0.92)],
            "loss": torch.tensor([0.1, 0.3, 0.2]),
            "prediction": torch.tensor([1, 0, 1]),
        },
        # Test case 2: PyTorch tensors
        {
            "embeddings": torch.randn(3, 10),
            "logits": [
                torch.tensor([0.1, 0.9]),
                torch.tensor([0.8, 0.2]),
                torch.tensor([0.3, 0.7]),
            ],
        },
        # Test case 3: NumPy arrays
        {
            "features": torch.randn(3, 5, 5),
            "scores": [
                torch.tensor([1.0, 2.0]),
                torch.tensor([1.5, 2.5]),
                torch.tensor([0.5, 3.0]),
            ],
        },
        # Test case 4: Mixed types
        {
            "numbers": [42, 17, 99],
            "tensors": [torch.ones(2), torch.zeros(2), torch.full((2,), 0.5)],
            "tensors2": [
                torch.tensor([1, 2]),
                torch.tensor([3, 4]),
                torch.tensor([5, 6]),
            ],
        },
        # Test case 5: Unsupported types
        {
            "strings": [  # unsupported type
                "string1",
                "string2",
                "string3",
            ]
        },
        # Test case 6: Mixed supported types
        {
            "tensor": torch.randn(3, 1, 1),
            "tensor2": torch.randn(3, 3, 1, 1),
            "tensor3": torch.randn(3, 1, 1),
        },
    ],
)
def test_batch_unbatch_roundtrip(
    sample_ids: list[str], test_data: dict[str, list[Any]]
):
    """Test that batch -> unbatch preserves all data."""
    # Create individual samples
    samples = []
    for i, sample_id in enumerate(sample_ids):
        sample_data = {key: values[i] for key, values in test_data.items()}
        samples.append(SampleMetricData(sample_id=sample_id, data=sample_data))

    # Batch the samples
    batched = BatchMetricData.fromlist(samples)

    # Unbatch back to samples
    unbatched = batched.tolist()

    # Verify we get the same number of samples
    assert len(unbatched) == len(samples)

    # Verify each sample matches the original
    for original, recovered in zip(samples, unbatched, strict=True):
        assert original.sample_id == recovered.sample_id
        _assert_data_equal(original.data, recovered.data)


@pytest.mark.parametrize(
    "test_data",
    [
        {"lists": [[1, 2], [3, 4], [5, 6]]},
        {
            "tensors": [  # mixed tensor shapes
                torch.tensor([1, 2]),
                torch.tensor([[3, 4]]),
                torch.tensor([5, 6]),
            ]
        },
        {
            "mixed": [  # mixed types in a single key
                torch.tensor([1, 2]),
                torch.tensor([3, 4]),
                [5, 6],
            ]
        },
    ],
)
def test_invalid_data_throws_error(
    sample_ids: list[str], test_data: dict[str, list[Any]]
):
    """Test that batch -> unbatch preserves all data."""
    # Create individual samples
    with pytest.raises((ValidationError, ValueError)):
        # Create individual samples
        samples = []
        for i, sample_id in enumerate(sample_ids):
            sample_data = {key: values[i] for key, values in test_data.items()}
            samples.append(SampleMetricData(sample_id=sample_id, data=sample_data))

        # Batch the samples
        batched = BatchMetricData.fromlist(samples)

        # Unbatch back to samples
        unbatched = batched.tolist()

        # Verify we get the same number of samples
        assert len(unbatched) == len(samples)

        # Verify each sample matches the original
        for original, recovered in zip(samples, unbatched, strict=True):
            assert original.sample_id == recovered.sample_id
            _assert_data_equal(original.data, recovered.data)


def test_empty_batch():
    """Test handling of empty batch."""
    with pytest.raises(ValueError):
        BatchMetricData.fromlist([])


def test_single_sample_batch(sample_ids: list[str]):
    """Test batching/unbatching with single sample."""
    sample = SampleMetricData(
        sample_id=sample_ids[0], data={"value": 42, "tensor": torch.ones(3)}
    )

    batched = BatchMetricData.fromlist([sample])
    unbatched = batched.tolist()

    assert len(unbatched) == 1
    assert unbatched[0].sample_id == sample.sample_id
    _assert_data_equal(unbatched[0].data, sample.data)


def test_batch_properties(sample_ids: list[str]):
    """Test that batch properties are correctly set."""
    samples = [
        SampleMetricData(sample_id=sid, data={"value": i})
        for i, sid in enumerate(sample_ids)
    ]

    batched = BatchMetricData.fromlist(samples)

    assert batched.sample_id == sample_ids
    assert "value" in batched.data
    assert len(batched.data["value"]) == len(sample_ids)
