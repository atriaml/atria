import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from atria_insights.data_types._metric_data import BatchMetricData, SampleMetricData
from atria_insights.explainability_metrics._base import ExplainabilityMetricConfig
from atria_insights.storage.sample_cache_managers._metric_data_cacher import (
    MetricDataCacher,
)
from tests.utilities import _assert_data_equal


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_config():
    """Mock explainability metric config."""

    class MockConfig(ExplainabilityMetricConfig):
        type: str = "test_metric"

        @property
        def hash(self) -> str:
            return "test_hash_123"

    return MockConfig()


@pytest.fixture
def cacher(temp_cache_dir: Path, mock_config):
    """Create MetricDataCacher instance."""
    return MetricDataCacher(cache_dir=temp_cache_dir, config=mock_config)


@pytest.mark.parametrize(
    "test_data",
    [
        # Test case 1: Simple scalars only
        {"accuracy": 0.95, "loss": 0.1, "prediction": 1},
        # Test case 2: Tensors only
        {
            "embeddings": torch.randn(10),
            "logits": torch.tensor([0.1, 0.9]),
            "weights": torch.ones(5, 5),
        },
        # Test case 3: Mixed scalars and tensors
        {
            "accuracy": 0.87,
            "loss": torch.tensor(0.3),
            "embeddings": torch.randn(8),
            "label": 2,
        },
        # Test case 4: Different tensor types and shapes
        {
            "scalar_tensor": torch.tensor(42.0),
            "vector": torch.arange(5),
            "matrix": torch.eye(3),
            "3d_tensor": torch.randn(2, 3, 4),
            "float_value": 3.14,
            "int_value": 100,
        },
        # Test case 5: NumPy arrays (should be converted to tensors or scalars)
        {
            "np_scalar": np.float32(1.5),
            "np_array": torch.tensor([1, 2, 3]),
            "tensor": torch.tensor([4, 5, 6]),
            "regular_scalar": 42,
        },
    ],
)
def test_cache_load_roundtrip(cacher: MetricDataCacher, test_data: dict[str, Any]):
    """Test that cache -> load preserves all data."""
    sample_id = "test_sample_001"

    # Create sample metric data
    original_sample = SampleMetricData(sample_id=sample_id, data=test_data)

    # Cache the sample
    cacher.save_sample(original_sample)

    # Verify sample exists in cache
    assert cacher.sample_exists(sample_id), f"Sample {sample_id} should exist in cache"

    # Load the sample back
    loaded_sample = cacher.load_sample(sample_id)

    # Verify the loaded sample matches original
    assert loaded_sample.sample_id == original_sample.sample_id

    _assert_data_equal(loaded_sample.data, original_sample.data)


def test_multiple_samples_caching(cacher: MetricDataCacher):
    """Test caching multiple samples."""
    # first we create multiple sample metric data
    sample_data = [
        SampleMetricData(
            sample_id=f"sample_{i:03d}",
            data={
                "value": i * 10,
                "tensor": torch.randn(3),
                "accuracy": 0.9 + i * 0.01,
            },
        )
        for i in range(5)
    ]

    # Cache all samples
    for data in sample_data:
        cacher.save_sample(data)

    # Verify all samples exist
    for data in sample_data:
        assert cacher.sample_exists(data.sample_id)

    # Load and verify all samples
    loaded_samples = []
    for original in sample_data:
        loaded = cacher.load_sample(original.sample_id)
        assert loaded.sample_id == original.sample_id
        _assert_data_equal(loaded.data, original.data)
        loaded_samples.append(loaded)


def test_multiple_samples_caching_with_batch_unbatch(cacher: MetricDataCacher):
    """Test caching multiple samples."""
    # first we create multiple sample metric data
    batch_metric_data = BatchMetricData(
        sample_id=[f"sample_{i:03d}" for i in range(5)],
        data={
            "value": [i * 10 for i in range(5)],
            "tensor": torch.randn(5, 1, 1),
            "tensor2": torch.randn(5, 3, 1, 1),
            "tensor3": torch.randn(5, 1, 1),
            "accuracy": [0.9 + i * 0.01 for i in range(5)],
        },
    )

    # Cache all samples
    for data in batch_metric_data.tolist():
        cacher.save_sample(data)

    # Load and verify all samples
    loaded_sample_metric_data = []
    for sample_id in batch_metric_data.sample_id:
        loaded = cacher.load_sample(sample_id)
        assert loaded.sample_id == sample_id
        loaded_sample_metric_data.append(loaded)

    loaded_batch_metric_data = BatchMetricData.fromlist(loaded_sample_metric_data)
    _assert_data_equal(loaded_batch_metric_data.data, batch_metric_data.data)


def test_batch_unbatch_roundtrip_simple(cacher: MetricDataCacher):
    """Test that batch -> unbatch preserves all data."""
    # Create individual samples
    batch_metric_data = BatchMetricData(
        sample_id=[f"sample_{i:03d}" for i in range(3)],
        data={
            "completeness": [0, 1, 0],
            "completeness_tensor": torch.tensor([0, 1, 0]),
        },
    )

    # Cache all samples
    for data in batch_metric_data.tolist():
        cacher.save_sample(data)

    # Load and verify all samples
    loaded_sample_metric_data = []
    for sample_id in batch_metric_data.sample_id:
        loaded = cacher.load_sample(sample_id)
        assert loaded.sample_id == sample_id
        loaded_sample_metric_data.append(loaded)

    loaded_batch_metric_data = BatchMetricData.fromlist(loaded_sample_metric_data)
    _assert_data_equal(loaded_batch_metric_data.data, batch_metric_data.data)


def test_overwrite_existing_sample(cacher: MetricDataCacher):
    """Test overwriting existing cached sample."""
    sample_id = "overwrite_test"

    # Cache initial data
    original_data = SampleMetricData(
        sample_id=sample_id, data={"value": 1, "tensor": torch.ones(2)}
    )
    cacher.save_sample(original_data)

    # Cache updated data with same sample_id
    updated_data = SampleMetricData(
        sample_id=sample_id,
        data={"value": 2, "tensor": torch.zeros(2), "new_field": 42},
    )
    cacher.save_sample(updated_data)

    # Load and verify we get the updated data
    loaded = cacher.load_sample(sample_id)
    assert loaded.sample_id == sample_id
    _assert_data_equal(loaded.data, updated_data.data)


def test_nonexistent_sample(cacher: MetricDataCacher):
    """Test loading non-existent sample raises appropriate error."""
    with pytest.raises((KeyError, FileNotFoundError, ValueError)):
        cacher.load_sample("nonexistent_sample")


def test_cache_persistence_across_instances(temp_cache_dir: Path, mock_config):
    """Test that cache persists across different cacher instances."""
    sample_id = "persistence_test"
    sample_data = SampleMetricData(
        sample_id=sample_id, data={"value": 99, "tensor": torch.tensor([1.0, 2.0, 3.0])}
    )

    # Cache with first instance
    cacher1 = MetricDataCacher(cache_dir=temp_cache_dir, config=mock_config)
    cacher1.save_sample(sample_data)

    # Load with second instance
    cacher2 = MetricDataCacher(cache_dir=temp_cache_dir, config=mock_config)
    assert cacher2.sample_exists(sample_id)
    loaded_sample = cacher2.load_sample(sample_id)

    # Verify data integrity
    assert loaded_sample.sample_id == sample_data.sample_id
    _assert_data_equal(loaded_sample.data, sample_data.data)


def test_empty_data_caching(cacher: MetricDataCacher):
    """Test caching sample with empty data."""
    sample_id = "empty_data_test"
    empty_sample = SampleMetricData(sample_id=sample_id, data={})

    cacher.save_sample(empty_sample)
    loaded_sample = cacher.load_sample(sample_id)

    assert loaded_sample.sample_id == sample_id
    assert loaded_sample.data == {}


def test_cache_file_structure(cacher: MetricDataCacher, temp_cache_dir: Path):
    """Test that cache creates expected file structure."""
    sample_id = "file_structure_test"
    sample_data = SampleMetricData(sample_id=sample_id, data={"value": 1})

    cacher.save_sample(sample_data)

    # Check that cache file was created
    expected_file = temp_cache_dir / "metrics" / "test_metric-test_hash_123.hdf5"
    assert expected_file.exists(), f"Expected cache file {expected_file} does not exist"


@pytest.mark.parametrize(
    "invalid_data",
    [
        # Test case 1: Unsupported data types
        {"unsupported": object()},
        {"complex_obj": {"nested": "dict"}},
        # Add more invalid cases based on your data model validation
    ],
)
def test_invalid_data_handling(cacher: MetricDataCacher, invalid_data: dict[str, Any]):
    """Test that invalid data types are handled appropriately."""
    sample_id = "invalid_test"

    with pytest.raises(ValidationError):
        invalid_sample = SampleMetricData(sample_id=sample_id, data=invalid_data)
        cacher.save_sample(invalid_sample)
