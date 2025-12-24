import pytest
import torch
from pydantic import ValidationError

from atria_insights.data_types._features import BatchFeatures, SampleFeatures


@pytest.fixture
def sample_ids() -> list[str]:
    """Sample IDs for testing."""
    return ["sample_001", "sample_002", "sample_003"]


@pytest.mark.parametrize(
    "feature_data",
    [
        # Test case 1: Simple 1D features
        {
            "conv1": [torch.randn(64), torch.randn(64), torch.randn(64)],
            "conv2": [torch.randn(128), torch.randn(128), torch.randn(128)],
        },
        # Test case 2: 2D feature maps
        {
            "layer1": [torch.randn(32, 32), torch.randn(32, 32), torch.randn(32, 32)],
            "layer2": [torch.randn(16, 16), torch.randn(16, 16), torch.randn(16, 16)],
        },
        # Test case 3: 3D features (e.g., from conv layers)
        {
            "conv_features": [
                torch.randn(64, 14, 14),
                torch.randn(64, 14, 14),
                torch.randn(64, 14, 14),
            ],
            "pool_features": [
                torch.randn(128, 7, 7),
                torch.randn(128, 7, 7),
                torch.randn(128, 7, 7),
            ],
        },
        # Test case 4: Mixed dimensional features
        {
            "embedding": [torch.randn(256), torch.randn(256), torch.randn(256)],
            "attention": [
                torch.randn(12, 64),
                torch.randn(12, 64),
                torch.randn(12, 64),
            ],
            "conv": [
                torch.randn(32, 8, 8),
                torch.randn(32, 8, 8),
                torch.randn(32, 8, 8),
            ],
        },
        # Test case 5: Single feature layer
        {"final_layer": [torch.randn(10), torch.randn(10), torch.randn(10)]},
        # Test case 6: Large feature tensors
        {
            "backbone": [
                torch.randn(512, 14, 14),
                torch.randn(512, 14, 14),
                torch.randn(512, 14, 14),
            ]
        },
    ],
)
def test_batch_unbatch_roundtrip(
    sample_ids: list[str], feature_data: dict[str, list[torch.Tensor]]
):
    """Test that batch -> unbatch preserves all feature data."""
    # Create individual samples
    samples = []
    for i, sample_id in enumerate(sample_ids):
        feature_keys = tuple(feature_data.keys())
        features = tuple(feature_data[key][i] for key in feature_keys)

        samples.append(
            SampleFeatures(
                sample_id=sample_id, feature_keys=feature_keys, features=features
            )
        )

    # Batch the samples
    batched = BatchFeatures.fromlist(samples)

    # Unbatch back to samples
    unbatched = batched.tolist()

    # Verify we get the same number of samples
    assert len(unbatched) == len(samples)
    assert len(unbatched) == len(sample_ids)

    # Verify each sample matches the original
    for original, recovered in zip(samples, unbatched, strict=True):
        assert original.sample_id == recovered.sample_id
        assert original.feature_keys == recovered.feature_keys
        assert len(original.features) == len(recovered.features)

        # Check each feature tensor
        for orig_feat, recv_feat in zip(
            original.features, recovered.features, strict=True
        ):
            assert torch.equal(orig_feat, recv_feat), (
                "Feature tensors should be identical"
            )


@pytest.mark.parametrize(
    "invalid_data",
    [
        # Test case 1: Mismatched feature shapes
        {
            "layer1": [
                torch.randn(64),
                torch.randn(128),
                torch.randn(64),
            ]  # Different sizes
        }
        # Test case 2: Inconsistent feature keys across samples
        # This would be caught at the SampleFeatures level, not BatchFeatures.fromlist
    ],
)
def test_invalid_data_throws_error(
    sample_ids: list[str], invalid_data: dict[str, list[torch.Tensor]]
):
    """Test that incompatible feature data throws appropriate errors."""
    with pytest.raises((RuntimeError, ValueError)):
        samples = []
        for i, sample_id in enumerate(sample_ids):
            feature_keys = tuple(invalid_data.keys())
            features = tuple(invalid_data[key][i] for key in feature_keys)

            samples.append(
                SampleFeatures(
                    sample_id=sample_id, feature_keys=feature_keys, features=features
                )
            )

        # This should fail due to tensor shape mismatch
        BatchFeatures.fromlist(samples)


def test_empty_features():
    """Test caching sample with no features."""
    with pytest.raises(ValidationError):
        sample_id = "empty_features_test"
        SampleFeatures(sample_id=sample_id, feature_keys=(), features=())


def test_empty_batch():
    """Test handling of empty batch."""
    with pytest.raises(ValueError, match="data list is empty"):
        BatchFeatures.fromlist([])


def test_single_sample_batch(sample_ids: list[str]):
    """Test batching/unbatching with single sample."""
    sample = SampleFeatures(
        sample_id=sample_ids[0],
        feature_keys=("conv1", "fc"),
        features=(torch.ones(32), torch.zeros(10)),
    )

    batched = BatchFeatures.fromlist([sample])
    unbatched = batched.tolist()

    assert len(unbatched) == 1
    assert unbatched[0].sample_id == sample.sample_id
    assert unbatched[0].feature_keys == sample.feature_keys
    for orig, recv in zip(sample.features, unbatched[0].features, strict=True):
        assert torch.equal(orig, recv)


def test_batch_properties(sample_ids: list[str]):
    """Test that batch properties are correctly set."""
    feature_keys = ("layer1", "layer2")
    samples = [
        SampleFeatures(
            sample_id=sid,
            feature_keys=feature_keys,
            features=(torch.randn(16), torch.randn(32)),
        )
        for sid in sample_ids
    ]

    batched = BatchFeatures.fromlist(samples)

    assert batched.sample_id == sample_ids
    assert batched.feature_keys == feature_keys
    assert len(batched.features) == len(feature_keys)

    # Check that each batched feature has correct batch dimension
    for feature_tensor in batched.features:
        assert feature_tensor.size(0) == len(sample_ids)


def test_as_ordered_dict(sample_ids: list[str]):
    """Test conversion to OrderedDict."""
    feature_keys = ("conv1", "conv2", "fc")
    samples = [
        SampleFeatures(
            sample_id=sid,
            feature_keys=feature_keys,
            features=(torch.randn(64), torch.randn(128), torch.randn(10)),
        )
        for sid in sample_ids
    ]

    batched = BatchFeatures.fromlist(samples)
    ordered_dict = batched.as_ordered_dict()

    assert list(ordered_dict.keys()) == list(feature_keys)
    assert len(ordered_dict) == len(feature_keys)

    # Verify tensors match
    for key, tensor in zip(feature_keys, batched.features, strict=True):
        assert torch.equal(ordered_dict[key], tensor)


def test_consistent_feature_keys_requirement(sample_ids: list[str]):
    """Test that all samples must have the same feature keys."""
    # Create samples with different feature keys
    sample1 = SampleFeatures(
        sample_id=sample_ids[0],
        feature_keys=("layer1", "layer2"),
        features=(torch.randn(16), torch.randn(32)),
    )
    sample2 = SampleFeatures(
        sample_id=sample_ids[1],
        feature_keys=("layer1", "layer3"),  # Different key
        features=(torch.randn(16), torch.randn(32)),
    )

    # This should work fine at the individual level
    assert sample1.feature_keys != sample2.feature_keys

    # But batching should handle this appropriately
    # (The current implementation assumes consistent keys)
    with pytest.raises((ValueError, KeyError)):
        BatchFeatures.fromlist([sample1, sample2])


def test_feature_tensor_stacking(sample_ids: list[str]):
    """Test that feature tensors are properly stacked."""
    # Create samples with known values
    samples = [
        SampleFeatures(
            sample_id=sample_ids[i],
            feature_keys=("test_feature",),
            features=(torch.full((3,), float(i)),),  # Fill with sample index
        )
        for i in range(len(sample_ids))
    ]

    batched = BatchFeatures.fromlist(samples)

    # Check that stacking worked correctly
    expected_tensor = torch.stack(
        [torch.full((3,), float(i)) for i in range(len(sample_ids))]
    )
    assert torch.equal(batched.features[0], expected_tensor)
