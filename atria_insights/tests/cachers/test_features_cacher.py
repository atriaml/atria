import shutil
import tempfile
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from atria_insights.data_types._features import BatchFeatures, SampleFeatures
from atria_insights.storage.sample_cache_managers._features_cacher import FeaturesCacher


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def cacher(temp_cache_dir: Path):
    """Create FeaturesCacher instance."""
    return FeaturesCacher(cache_dir=temp_cache_dir)


@pytest.mark.parametrize(
    "feature_data",
    [
        # Test case 1: Simple 1D features
        {"keys": ("conv1", "conv2"), "features": (torch.randn(64), torch.randn(128))},
        # Test case 2: 2D feature maps
        {
            "keys": ("layer1", "layer2"),
            "features": (torch.randn(32, 32), torch.randn(16, 16)),
        },
        # Test case 3: 3D features (conv layers)
        {
            "keys": ("conv_features", "pool_features"),
            "features": (torch.randn(64, 14, 14), torch.randn(128, 7, 7)),
        },
        # Test case 4: Mixed dimensional features
        {
            "keys": ("embedding", "attention", "conv"),
            "features": (torch.randn(256), torch.randn(12, 64), torch.randn(32, 8, 8)),
        },
        # Test case 5: Single feature
        {"keys": ("final_layer",), "features": (torch.randn(10),)},
        # Test case 6: Large feature tensors
        {"keys": ("backbone",), "features": (torch.randn(512, 14, 14),)},
    ],
)
def test_cache_load_roundtrip(cacher: FeaturesCacher, feature_data: dict[str, tuple]):
    """Test that cache -> load preserves all feature data."""
    sample_id = "test_sample_001"

    # Create sample features
    original_sample = SampleFeatures(
        sample_id=sample_id,
        feature_keys=feature_data["keys"],
        features=feature_data["features"],
    )

    # Cache the sample
    cacher.save_sample(original_sample)

    # Verify sample exists in cache
    assert cacher.sample_exists(sample_id), f"Sample {sample_id} should exist in cache"

    # Load the sample back
    loaded_sample = cacher.load_sample(sample_id)

    # Verify the loaded sample matches original
    assert loaded_sample.sample_id == original_sample.sample_id
    assert loaded_sample.feature_keys == original_sample.feature_keys
    assert len(loaded_sample.features) == len(original_sample.features)

    # Check each feature tensor
    for orig_feat, loaded_feat in zip(
        original_sample.features, loaded_sample.features, strict=True
    ):
        assert torch.equal(orig_feat, loaded_feat), (
            "Feature tensors should be identical"
        )


def test_multiple_samples_caching(cacher: FeaturesCacher):
    """Test caching multiple samples."""
    # Create multiple sample features
    samples = [
        SampleFeatures(
            sample_id=f"sample_{i:03d}",
            feature_keys=("layer1", "layer2", "fc"),
            features=(
                torch.randn(32) * i,  # Make each sample unique
                torch.randn(64) * i,
                torch.randn(10) * i,
            ),
        )
        for i in range(1, 6)  # Start from 1 to avoid zero multiplication
    ]

    # Cache all samples
    for sample in samples:
        cacher.save_sample(sample)

    # Verify all samples exist
    for sample in samples:
        assert cacher.sample_exists(sample.sample_id)

    # Load and verify all samples
    for original in samples:
        loaded = cacher.load_sample(original.sample_id)
        assert loaded.sample_id == original.sample_id
        assert loaded.feature_keys == original.feature_keys

        for orig_feat, loaded_feat in zip(
            original.features, loaded.features, strict=True
        ):
            assert torch.equal(orig_feat, loaded_feat)


def test_multiple_samples_caching_with_batch_unbatch(cacher: FeaturesCacher):
    """Test caching multiple samples using batch operations."""
    # Create batch features
    batch_features = BatchFeatures(
        sample_id=[f"batch_sample_{i:03d}" for i in range(3)],
        feature_keys=("conv1", "conv2", "fc"),
        features=(
            torch.randn(3, 64),  # 3 samples, 64 features each
            torch.randn(3, 128),  # 3 samples, 128 features each
            torch.randn(3, 10),  # 3 samples, 10 features each
        ),
    )

    # Cache all samples from batch
    for sample in batch_features.tolist():
        cacher.save_sample(sample)

    # Load and verify all samples
    loaded_samples = []
    for sample_id in batch_features.sample_id:
        loaded = cacher.load_sample(sample_id)
        assert loaded.sample_id == sample_id
        loaded_samples.append(loaded)

    # Reconstruct batch and verify
    loaded_batch = BatchFeatures.fromlist(loaded_samples)

    assert loaded_batch.sample_id == batch_features.sample_id
    assert loaded_batch.feature_keys == batch_features.feature_keys

    for orig_feat, loaded_feat in zip(
        batch_features.features, loaded_batch.features, strict=True
    ):
        assert torch.equal(orig_feat, loaded_feat)


def test_batch_unbatch_roundtrip_simple(cacher: FeaturesCacher):
    """Test simple batch -> unbatch -> cache -> load roundtrip."""
    # Create simple batch features
    batch_features = BatchFeatures(
        sample_id=["simple_001", "simple_002", "simple_003"],
        feature_keys=("feature1", "feature2"),
        features=(
            torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),  # 3x2
            torch.tensor([[0.1], [0.2], [0.3]]),  # 3x1
        ),
    )

    # Cache all samples from batch
    for sample in batch_features.tolist():
        cacher.save_sample(sample)

    # Load and verify all samples
    loaded_samples = []
    for sample_id in batch_features.sample_id:
        loaded = cacher.load_sample(sample_id)
        assert loaded.sample_id == sample_id
        loaded_samples.append(loaded)

    # Reconstruct batch and verify
    loaded_batch = BatchFeatures.fromlist(loaded_samples)

    for orig_feat, loaded_feat in zip(
        batch_features.features, loaded_batch.features, strict=True
    ):
        assert torch.equal(orig_feat, loaded_feat)


def test_overwrite_existing_sample(cacher: FeaturesCacher):
    """Test overwriting existing cached sample."""
    sample_id = "overwrite_test"

    # Cache initial data
    original_data = SampleFeatures(
        sample_id=sample_id, feature_keys=("layer1",), features=(torch.ones(5),)
    )
    cacher.save_sample(original_data)

    # Cache updated data with same sample_id
    updated_data = SampleFeatures(
        sample_id=sample_id,
        feature_keys=("layer1", "layer2"),  # Different structure
        features=(torch.zeros(5), torch.full((3,), 2.0)),
    )
    cacher.save_sample(updated_data)

    # Load and verify we get the updated data
    loaded = cacher.load_sample(sample_id)
    assert loaded.sample_id == sample_id
    assert loaded.feature_keys == updated_data.feature_keys

    for updated_feat, loaded_feat in zip(
        updated_data.features, loaded.features, strict=True
    ):
        assert torch.equal(updated_feat, loaded_feat)


def test_nonexistent_sample(cacher: FeaturesCacher):
    """Test loading non-existent sample raises appropriate error."""
    with pytest.raises((KeyError, FileNotFoundError, ValueError)):
        cacher.load_sample("nonexistent_sample")


def test_cache_persistence_across_instances(temp_cache_dir: Path):
    """Test that cache persists across different cacher instances."""
    sample_id = "persistence_test"
    sample_data = SampleFeatures(
        sample_id=sample_id,
        feature_keys=("test_feature",),
        features=(torch.tensor([1.0, 2.0, 3.0]),),
    )

    # Cache with first instance
    cacher1 = FeaturesCacher(cache_dir=temp_cache_dir)
    cacher1.save_sample(sample_data)

    # Load with second instance
    cacher2 = FeaturesCacher(cache_dir=temp_cache_dir)
    assert cacher2.sample_exists(sample_id)
    loaded_sample = cacher2.load_sample(sample_id)

    # Verify data integrity
    assert loaded_sample.sample_id == sample_data.sample_id
    assert loaded_sample.feature_keys == sample_data.feature_keys

    for orig_feat, loaded_feat in zip(
        sample_data.features, loaded_sample.features, strict=True
    ):
        assert torch.equal(orig_feat, loaded_feat)


def test_empty_features_caching(cacher: FeaturesCacher):
    """Test caching sample with no features."""
    with pytest.raises(ValidationError):
        sample_id = "empty_features_test"
        SampleFeatures(sample_id=sample_id, feature_keys=(), features=())


def test_cache_file_structure(cacher: FeaturesCacher, temp_cache_dir: Path):
    """Test that cache creates expected file structure."""
    sample_id = "file_structure_test"
    sample_data = SampleFeatures(
        sample_id=sample_id, feature_keys=("test",), features=(torch.tensor([1.0]),)
    )

    cacher.save_sample(sample_data)

    # Check that cache file was created (adjust path based on your implementation)
    expected_file = temp_cache_dir / "features.hdf5"
    assert expected_file.exists(), f"Expected cache file {expected_file} does not exist"


def test_large_feature_tensors(cacher: FeaturesCacher):
    """Test caching large feature tensors."""
    sample_id = "large_tensor_test"
    large_sample = SampleFeatures(
        sample_id=sample_id,
        feature_keys=("large_conv", "large_fc"),
        features=(
            torch.randn(512, 28, 28),  # Large conv feature
            torch.randn(4096),  # Large FC feature
        ),
    )

    cacher.save_sample(large_sample)
    loaded_sample = cacher.load_sample(sample_id)

    assert loaded_sample.sample_id == sample_id
    assert loaded_sample.feature_keys == large_sample.feature_keys

    for orig_feat, loaded_feat in zip(
        large_sample.features, loaded_sample.features, strict=True
    ):
        assert torch.equal(orig_feat, loaded_feat)


@pytest.mark.parametrize(
    "invalid_features",
    [
        # Test case 1: Mismatched keys and features length
        {
            "keys": ("layer1", "layer2"),
            "features": (torch.randn(10),),  # Only one feature for two keys
        }
    ],
)
def test_invalid_data_handling(cacher: FeaturesCacher, invalid_features: dict):
    """Test that invalid feature data is handled appropriately."""
    sample_id = "invalid_test"

    with pytest.raises((ValidationError, ValueError)):
        invalid_sample = SampleFeatures(
            sample_id=sample_id,
            feature_keys=invalid_features["keys"],
            features=invalid_features["features"],
        )
        cacher.save_sample(invalid_sample)
