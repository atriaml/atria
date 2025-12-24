import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
from atria_registry._module_base import ModuleConfig
from pydantic import BaseModel

from atria_insights.data_types._explanation_state import (
    BatchExplanationState,
    MultiTargetBatchExplanation,
    MultiTargetSampleExplanation,
    SampleExplanation,
    SampleExplanationState,
    SampleExplanationTarget,
)
from atria_insights.storage.sample_cache_managers._explanation_state import (
    ExplanationStateCacher,
)


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_config():
    """Mock explainable model pipeline config."""

    class MockExplainerConfig(BaseModel):
        type: str = "test_explainer"

    class MockConfig(ModuleConfig):
        explainer: MockExplainerConfig = MockExplainerConfig()

        @property
        def hash(self) -> str:
            return "test_hash_789"

    return MockConfig()


@pytest.fixture
def cacher(temp_cache_dir: Path, mock_config):
    """Create ExplanationStateCacher instance."""
    return ExplanationStateCacher(cache_dir=temp_cache_dir, config=mock_config)


@pytest.mark.parametrize(
    "explanation_data",
    [
        # Test case 1: Simple 1D explanations
        {
            "feature_keys": ("conv1", "conv2"),
            "explanations": SampleExplanation(
                value=(torch.randn(1, 64), torch.randn(1, 128))
            ),
            "model_outputs": torch.randn(1, 10),
            "frozen_features": None,
            "target": SampleExplanationTarget(value=1, name="target_1"),
            "is_multitarget": False,
        },
        # Test case 2: 2D explanation maps
        {
            "feature_keys": ("layer1", "layer2"),
            "explanations": SampleExplanation(
                value=(torch.randn(1, 32, 32), torch.randn(1, 16, 16))
            ),
            "model_outputs": torch.randn(1, 5),
            "frozen_features": torch.randn(64),
            "target": SampleExplanationTarget(value=0, name="target_0"),
            "is_multitarget": False,
        },
        # Test case 3: 3D explanations (conv layers)
        {
            "feature_keys": ("conv_features", "pool_features"),
            "explanations": SampleExplanation(
                value=(torch.randn(1, 64, 14, 14), torch.randn(1, 128, 7, 7))
            ),
            "model_outputs": torch.randn(1, 1000),
            "frozen_features": None,
            "target": None,
            "is_multitarget": False,
        },
        # Test case 4: Mixed dimensional explanations
        {
            "feature_keys": ("embedding", "attention", "conv"),
            "explanations": SampleExplanation(
                value=(
                    torch.randn(1, 256),
                    torch.randn(1, 12, 64),
                    torch.randn(1, 32, 8, 8),
                )
            ),
            "model_outputs": torch.randn(1, 2),
            "frozen_features": torch.randn(100),
            "target": SampleExplanationTarget(value=1, name="target_1"),
            "is_multitarget": False,
        },
        # Test case 5: Single explanation layer
        {
            "feature_keys": ("final_layer",),
            "explanations": SampleExplanation(value=(torch.randn(1, 10),)),
            "model_outputs": torch.randn(1, 10),
            "frozen_features": None,
            "target": SampleExplanationTarget(value=2, name="target_2"),
            "is_multitarget": False,
        },
    ],
)
def test_single_sample_cache_roundtrip(
    cacher: ExplanationStateCacher, explanation_data: dict[str, Any]
):
    """Test that cache -> load preserves all explanation state data."""
    sample_id = "test_sample_001"

    # Create sample explanation state
    original_sample = SampleExplanationState(
        sample_id=sample_id,
        target=explanation_data["target"],
        feature_keys=explanation_data["feature_keys"],
        frozen_features=explanation_data["frozen_features"],
        explanations=explanation_data["explanations"],
        model_outputs=explanation_data["model_outputs"],
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
    assert loaded_sample.is_multitarget == original_sample.is_multitarget

    # Check target
    if original_sample.target is not None:
        assert loaded_sample.target is not None
        if isinstance(original_sample.target, list):
            assert loaded_sample.target == original_sample.target
        else:
            assert original_sample.target == loaded_sample.target
    else:
        assert loaded_sample.target is None

    # Check model outputs
    assert torch.equal(original_sample.model_outputs, loaded_sample.model_outputs)

    # Check frozen features
    if original_sample.frozen_features is not None:
        assert loaded_sample.frozen_features is not None
        assert torch.equal(
            original_sample.frozen_features, loaded_sample.frozen_features
        )
    else:
        assert loaded_sample.frozen_features is None

    # Check explanations
    if isinstance(original_sample.explanations, SampleExplanation):
        assert isinstance(loaded_sample.explanations, SampleExplanation)
        for orig_exp, loaded_exp in zip(
            original_sample.explanations.value,
            loaded_sample.explanations.value,
            strict=True,
        ):
            assert torch.equal(orig_exp, loaded_exp), (
                "Explanation tensors should be identical"
            )
    else:
        assert isinstance(loaded_sample.explanations, MultiTargetSampleExplanation)
        assert len(original_sample.explanations.value) == len(
            loaded_sample.explanations.value
        )
        for orig_exp_set, loaded_exp_set in zip(
            original_sample.explanations.value,
            loaded_sample.explanations.value,
            strict=True,
        ):
            for orig_exp, loaded_exp in zip(
                orig_exp_set.value, loaded_exp_set.value, strict=True
            ):
                assert torch.equal(orig_exp, loaded_exp)


def test_multi_sample_cache_roundtrip(cacher: ExplanationStateCacher):
    """Test caching multiple explanation state samples."""
    samples = []
    for i in range(5):
        sample = SampleExplanationState(
            sample_id=f"sample_{i:03d}",
            target=SampleExplanationTarget(value=1, name=f"target_{i}"),
            feature_keys=("layer1", "layer2", "fc"),
            frozen_features=None,
            explanations=SampleExplanation(
                value=(
                    torch.randn(1, 32) * i,  # Make each sample unique
                    torch.randn(1, 64) * i,
                    torch.randn(1, 10) * i,
                )
            ),
            model_outputs=torch.randn(1, 5) * i,
        )
        samples.append(sample)

    # batch and unbatch
    samples = BatchExplanationState.fromlist(samples).tolist()

    # Cache all samples
    for sample in samples:
        cacher.save_sample(sample)

    # Verify all samples exist
    for sample in samples:
        assert cacher.sample_exists(sample.sample_id)

    # Load and verify all samples
    loaded_samples = []
    for original_sample in samples:
        loaded_sample = cacher.load_sample(original_sample.sample_id)
        assert loaded_sample.sample_id == original_sample.sample_id
        assert loaded_sample.feature_keys == original_sample.feature_keys

        # Verify the loaded sample matches original
        assert loaded_sample.sample_id == original_sample.sample_id
        assert loaded_sample.feature_keys == original_sample.feature_keys
        assert loaded_sample.is_multitarget == original_sample.is_multitarget

        # Check target
        if original_sample.target is not None:
            assert loaded_sample.target is not None
            if isinstance(original_sample.target, list):
                assert loaded_sample.target == original_sample.target
            else:
                assert original_sample.target == loaded_sample.target
        else:
            assert loaded_sample.target is None

        # Check model outputs
        assert torch.equal(original_sample.model_outputs, loaded_sample.model_outputs)

        # Check frozen features
        if original_sample.frozen_features is not None:
            assert loaded_sample.frozen_features is not None
            assert torch.equal(
                original_sample.frozen_features, loaded_sample.frozen_features
            )
        else:
            assert loaded_sample.frozen_features is None

        # Check explanations
        if isinstance(original_sample.explanations, SampleExplanation):
            assert isinstance(loaded_sample.explanations, SampleExplanation)
            for orig_exp, loaded_exp in zip(
                original_sample.explanations.value,
                loaded_sample.explanations.value,
                strict=True,
            ):
                assert torch.equal(orig_exp, loaded_exp), (
                    "Explanation tensors should be identical"
                )
        else:
            assert isinstance(loaded_sample.explanations, MultiTargetSampleExplanation)
            assert len(original_sample.explanations.value) == len(
                loaded_sample.explanations.value
            )
            for orig_exp_set, loaded_exp_set in zip(
                original_sample.explanations.value,
                loaded_sample.explanations.value,
                strict=True,
            ):
                for orig_exp, loaded_exp in zip(
                    orig_exp_set.value, loaded_exp_set.value, strict=True
                ):
                    assert torch.equal(orig_exp, loaded_exp)
        loaded_samples.append(loaded_sample)

    # Reconstruct batch and verify
    loaded_batch = BatchExplanationState.fromlist(loaded_samples)
    original_batch = BatchExplanationState.fromlist(samples)

    assert loaded_batch.sample_id == original_batch.sample_id
    assert loaded_batch.feature_keys == original_batch.feature_keys

    # Check explanations
    for orig_exp, loaded_exp in zip(
        original_batch.explanations.value, loaded_batch.explanations.value, strict=True
    ):
        assert torch.equal(orig_exp, loaded_exp)


def test_single_sample_multi_target_cache_roundtrip(cacher: ExplanationStateCacher):
    """Test caching multitarget explanation states."""
    sample_id = "multitarget_test"

    # Create multitarget sample
    targets = [
        SampleExplanationTarget(value=1, name="target_1"),
        SampleExplanationTarget(value=2, name="target_2"),
    ]
    explanations = MultiTargetSampleExplanation(
        value=[
            SampleExplanation(value=(torch.randn(1, 32), torch.randn(1, 64))),
            SampleExplanation(value=(torch.randn(1, 32), torch.randn(1, 64))),
        ]
    )

    original_sample = SampleExplanationState(
        sample_id=sample_id,
        target=targets,
        feature_keys=("layer1", "layer2"),
        frozen_features=None,
        explanations=explanations,
        model_outputs=torch.randn(1, 5),
    )

    # Cache the sample
    cacher.save_sample(original_sample)

    # Load the sample back
    loaded_sample = cacher.load_sample(sample_id)

    # Verify multitarget properties
    assert isinstance(original_sample.explanations, MultiTargetSampleExplanation)
    assert isinstance(loaded_sample.explanations, MultiTargetSampleExplanation)

    # Check targets
    assert isinstance(original_sample.target, list)
    assert isinstance(loaded_sample.target, list)
    assert len(loaded_sample.target) == 2
    assert loaded_sample.explanations.n_targets == 2
    for orig_tgt, loaded_tgt in zip(
        original_sample.target, loaded_sample.target, strict=True
    ):
        assert orig_tgt == loaded_tgt, "Targets should match"

    # Check explanations
    for orig_exp_set, loaded_exp_set in zip(
        original_sample.explanations.value,
        loaded_sample.explanations.value,
        strict=True,
    ):
        assert isinstance(orig_exp_set, SampleExplanation)
        assert isinstance(loaded_exp_set, SampleExplanation)
        for orig_exp, loaded_exp in zip(
            orig_exp_set.value, loaded_exp_set.value, strict=True
        ):
            assert torch.equal(orig_exp, loaded_exp)


def test_multi_sample_multi_target_cache_roundtrip(cacher: ExplanationStateCacher):
    """Test caching multiple explanation state samples."""
    samples = []
    for sample_idx in range(5):
        # Create multitarget sample
        targets = [
            SampleExplanationTarget(value=1, name="target_1"),
            SampleExplanationTarget(value=2, name="target_2"),
            SampleExplanationTarget(value=3, name="target_3"),
        ]
        explanations = MultiTargetSampleExplanation(
            value=[
                SampleExplanation(value=(torch.randn(1, 32), torch.randn(1, 64))),
                SampleExplanation(value=(torch.randn(1, 32), torch.randn(1, 64))),
                SampleExplanation(value=(torch.randn(1, 32), torch.randn(1, 64))),
            ]
        )

        sample = SampleExplanationState(
            sample_id=f"sample_{sample_idx:03d}",
            target=targets,
            feature_keys=("layer1", "layer2"),
            frozen_features=None,
            explanations=explanations,
            model_outputs=torch.randn(1, 5),
        )
        samples.append(sample)

    # batch and unbatch
    samples = BatchExplanationState.fromlist(samples).tolist()

    # Cache all samples
    for sample in samples:
        cacher.save_sample(sample)

    # Verify all samples exist
    for sample in samples:
        assert cacher.sample_exists(sample.sample_id)

    # Load and verify all samples
    loaded_samples = []
    for sample in samples:
        loaded_sample = cacher.load_sample(sample.sample_id)
        assert loaded_sample.sample_id == sample.sample_id
        assert loaded_sample.feature_keys == sample.feature_keys

        # Verify multitarget properties
        assert isinstance(sample.explanations, MultiTargetSampleExplanation)
        assert isinstance(loaded_sample.explanations, MultiTargetSampleExplanation)

        # Check targets
        assert isinstance(sample.target, list)
        assert isinstance(loaded_sample.target, list)
        assert len(loaded_sample.target) == 3
        assert loaded_sample.explanations.n_targets == 3
        for orig_tgt, loaded_tgt in zip(
            sample.target, loaded_sample.target, strict=True
        ):
            assert orig_tgt == loaded_tgt, "Targets should match"

        # Check explanations
        for orig_exp_set, loaded_exp_set in zip(
            sample.explanations.value, loaded_sample.explanations.value, strict=True
        ):
            assert isinstance(orig_exp_set, SampleExplanation)
            assert isinstance(loaded_exp_set, SampleExplanation)
            for orig_exp, loaded_exp in zip(
                orig_exp_set.value, loaded_exp_set.value, strict=True
            ):
                assert torch.equal(orig_exp, loaded_exp)
        loaded_samples.append(loaded_sample)

    # Reconstruct batch and verify
    loaded_batch = BatchExplanationState.fromlist(loaded_samples)
    original_batch = BatchExplanationState.fromlist(samples)

    assert loaded_batch.sample_id == original_batch.sample_id
    assert loaded_batch.feature_keys == original_batch.feature_keys

    # Check explanations
    # Verify multitarget properties
    assert isinstance(loaded_batch.explanations, MultiTargetBatchExplanation)
    assert isinstance(original_batch.explanations, MultiTargetBatchExplanation)
    for orig_exp_per_target, loaded_exp_per_target in zip(
        original_batch.explanations.value, loaded_batch.explanations.value, strict=True
    ):
        for orig_exp, loaded_exp in zip(
            orig_exp_per_target.value, loaded_exp_per_target.value, strict=True
        ):
            assert torch.equal(orig_exp, loaded_exp)


def test_overwrite_existing_sample(cacher: ExplanationStateCacher):
    """Test overwriting existing cached explanation state."""
    sample_id = "overwrite_test"

    # Cache initial data
    original_data = SampleExplanationState(
        sample_id=sample_id,
        target=SampleExplanationTarget(value=0, name="target_0"),
        feature_keys=("layer1",),
        frozen_features=None,
        explanations=SampleExplanation(value=(torch.ones(1, 5),)),
        model_outputs=torch.randn(1, 2),
    )
    cacher.save_sample(original_data)

    # Cache updated data with same sample_id
    updated_data = SampleExplanationState(
        sample_id=sample_id,
        target=SampleExplanationTarget(value=1, name="target_1"),
        feature_keys=("layer1", "layer2"),  # Different structure
        frozen_features=torch.randn(50),
        explanations=SampleExplanation(
            value=(torch.zeros(1, 5), torch.full((1, 3), 2.0))
        ),
        model_outputs=torch.randn(1, 2),
    )
    cacher.save_sample(updated_data)

    # Load and verify we get the updated data
    loaded = cacher.load_sample(sample_id)
    assert loaded.sample_id == sample_id
    assert loaded.feature_keys == updated_data.feature_keys

    # Check updated target
    assert updated_data.target == loaded.target

    # Check updated explanations
    for updated_exp, loaded_exp in zip(
        updated_data.explanations.value, loaded.explanations.value, strict=True
    ):
        assert torch.equal(updated_exp, loaded_exp)


def test_nonexistent_sample(cacher: ExplanationStateCacher):
    """Test loading non-existent sample raises appropriate error."""
    with pytest.raises((KeyError, FileNotFoundError, ValueError)):
        cacher.load_sample("nonexistent_sample")


def test_cache_persistence_across_instances(temp_cache_dir: Path, mock_config):
    """Test that cache persists across different cacher instances."""
    sample_id = "persistence_test"
    sample_data = SampleExplanationState(
        sample_id=sample_id,
        target=SampleExplanationTarget(value=2, name="target_2"),
        feature_keys=("test_feature",),
        frozen_features=None,
        explanations=SampleExplanation(value=(torch.tensor([[1.0, 2.0, 3.0]]),)),
        model_outputs=torch.tensor([[0.1, 0.9]]),
    )
    print("sample_data", sample_data)

    # Cache with first instance
    cacher1 = ExplanationStateCacher(cache_dir=temp_cache_dir, config=mock_config)
    cacher1.save_sample(sample_data)

    # Load with second instance
    cacher2 = ExplanationStateCacher(cache_dir=temp_cache_dir, config=mock_config)
    assert cacher2.sample_exists(sample_id)
    loaded_sample = cacher2.load_sample(sample_id)

    # Verify data integrity
    assert loaded_sample.sample_id == sample_data.sample_id
    assert loaded_sample.feature_keys == sample_data.feature_keys

    # Check target
    assert sample_data.target == loaded_sample.target

    # Check explanations per feature
    for orig_exp, loaded_exp in zip(
        sample_data.explanations.value, loaded_sample.explanations.value, strict=True
    ):
        assert torch.equal(orig_exp, loaded_exp)


def test_none_target_caching(cacher: ExplanationStateCacher):
    """Test caching explanation state with no target."""
    sample_id = "none_target_test"
    sample = SampleExplanationState(
        sample_id=sample_id,
        target=None,
        feature_keys=("layer1",),
        frozen_features=None,
        explanations=SampleExplanation(value=(torch.randn(1, 16),)),
        model_outputs=torch.randn(1, 5),
    )

    cacher.save_sample(sample)
    loaded_sample = cacher.load_sample(sample_id)

    assert loaded_sample.sample_id == sample_id
    assert loaded_sample.target is None
    assert torch.equal(sample.model_outputs, loaded_sample.model_outputs)


def test_cache_file_structure(cacher: ExplanationStateCacher, temp_cache_dir: Path):
    """Test that cache creates expected file structure."""
    sample_id = "file_structure_test"
    sample_data = SampleExplanationState(
        sample_id=sample_id,
        target=None,
        feature_keys=("test",),
        frozen_features=None,
        explanations=SampleExplanation(value=(torch.tensor([[1.0]]),)),
        model_outputs=torch.tensor([[0.5]]),
    )

    cacher.save_sample(sample_data)

    # Check that cache file was created
    expected_file = (
        temp_cache_dir / "test_explainer" / "explanations-test_hash_789.hdf5"
    )
    assert expected_file.exists(), f"Expected cache file {expected_file} does not exist"


def test_large_explanation_tensors(cacher: ExplanationStateCacher):
    """Test caching large explanation tensors."""
    sample_id = "large_tensor_test"
    large_sample = SampleExplanationState(
        sample_id=sample_id,
        target=SampleExplanationTarget(value=1, name="target_1"),
        feature_keys=("large_conv", "large_fc"),
        frozen_features=torch.randn(1000),
        explanations=SampleExplanation(
            value=(
                torch.randn(1, 512, 28, 28),  # Large conv explanation
                torch.randn(1, 4096),  # Large FC explanation
            )
        ),
        model_outputs=torch.randn(1, 10),
    )

    cacher.save_sample(large_sample)
    loaded_sample = cacher.load_sample(sample_id)

    assert loaded_sample.sample_id == sample_id
    assert loaded_sample.feature_keys == large_sample.feature_keys

    # Check large explanations
    for orig_exp, loaded_exp in zip(
        large_sample.explanations.value, loaded_sample.explanations.value, strict=True
    ):
        assert torch.equal(orig_exp, loaded_exp)

    # Check frozen features
    assert torch.equal(large_sample.frozen_features, loaded_sample.frozen_features)


@pytest.mark.parametrize(
    "invalid_data",
    [
        # Test case 1: Mismatched explanation and feature keys length
        {
            "feature_keys": ("layer1", "layer2"),
            "explanations": SampleExplanation(
                value=(torch.randn(1, 10),)
            ),  # Only one explanation for two keys
            "model_outputs": torch.randn(1, 5),
        }
    ],
)
def test_invalid_data_handling(cacher: ExplanationStateCacher, invalid_data: dict):
    """Test that invalid explanation data is handled appropriately."""
    sample_id = "invalid_test"

    with pytest.raises(AssertionError):
        invalid_sample = SampleExplanationState(
            sample_id=sample_id,
            target=None,
            feature_keys=invalid_data["feature_keys"],
            frozen_features=None,
            explanations=invalid_data["explanations"],
            model_outputs=invalid_data["model_outputs"],
        )
        cacher.save_sample(invalid_sample)
