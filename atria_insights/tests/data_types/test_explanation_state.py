from typing import Any

import pytest
import torch

from atria_insights.data_types._explanation_state import (
    BatchExplanation,
    BatchExplanationState,
    MultiTargetBatchExplanation,
    MultiTargetSampleExplanation,
    SampleExplanation,
    SampleExplanationState,
    SampleExplanationTarget,
)


class TestExplanationStateBatchingUnbatching:
    """Test that batching and unbatching preserves explanation state data integrity."""

    @pytest.fixture
    def sample_ids(self) -> list[str]:
        """Sample IDs for testing."""
        return ["sample_001", "sample_002", "sample_003"]

    @pytest.fixture
    def sample_targets(self, sample_ids: list[str]) -> list[SampleExplanationTarget]:
        """Sample targets for testing."""
        return [
            SampleExplanationTarget(value=i, name=f"target_{i}")
            for i in range(len(sample_ids))
        ]

    @pytest.mark.parametrize(
        "explanation_data",
        [
            # Test case 1: Simple 1D explanations
            {
                "feature_keys": ("conv1", "conv2"),
                "explanations": [
                    SampleExplanation(value=(torch.randn(1, 64), torch.randn(1, 128))),
                    SampleExplanation(value=(torch.randn(1, 64), torch.randn(1, 128))),
                    SampleExplanation(value=(torch.randn(1, 64), torch.randn(1, 128))),
                ],
                "model_outputs": [
                    torch.randn(1, 10),
                    torch.randn(1, 10),
                    torch.randn(1, 10),
                ],
                "frozen_features": [None, None, None],
            },
            # Test case 2: 2D explanation maps
            {
                "feature_keys": ("layer1", "layer2"),
                "explanations": [
                    SampleExplanation(
                        value=(torch.randn(1, 32, 32), torch.randn(1, 16, 16))
                    ),
                    SampleExplanation(
                        value=(torch.randn(1, 32, 32), torch.randn(1, 16, 16))
                    ),
                    SampleExplanation(
                        value=(torch.randn(1, 32, 32), torch.randn(1, 16, 16))
                    ),
                ],
                "model_outputs": [
                    torch.randn(1, 5),
                    torch.randn(1, 5),
                    torch.randn(1, 5),
                ],
                "frozen_features": [torch.randn(32), torch.randn(32), torch.randn(32)],
            },
            # Test case 3: 3D explanations (conv layers)
            {
                "feature_keys": ("conv_features", "pool_features"),
                "explanations": [
                    SampleExplanation(
                        value=(torch.randn(1, 64, 14, 14), torch.randn(1, 128, 7, 7))
                    ),
                    SampleExplanation(
                        value=(torch.randn(1, 64, 14, 14), torch.randn(1, 128, 7, 7))
                    ),
                    SampleExplanation(
                        value=(torch.randn(1, 64, 14, 14), torch.randn(1, 128, 7, 7))
                    ),
                ],
                "model_outputs": [
                    torch.randn(1, 1000),
                    torch.randn(1, 1000),
                    torch.randn(1, 1000),
                ],
                "frozen_features": [None, None, None],
            },
            # Test case 4: Mixed dimensional explanations
            {
                "feature_keys": ("embedding", "attention", "conv"),
                "explanations": [
                    SampleExplanation(
                        value=(
                            torch.randn(1, 256),
                            torch.randn(1, 12, 64),
                            torch.randn(1, 32, 8, 8),
                        )
                    ),
                    SampleExplanation(
                        value=(
                            torch.randn(1, 256),
                            torch.randn(1, 12, 64),
                            torch.randn(1, 32, 8, 8),
                        )
                    ),
                    SampleExplanation(
                        value=(
                            torch.randn(1, 256),
                            torch.randn(1, 12, 64),
                            torch.randn(1, 32, 8, 8),
                        )
                    ),
                ],
                "model_outputs": [
                    torch.randn(1, 2),
                    torch.randn(1, 2),
                    torch.randn(1, 2),
                ],
                "frozen_features": [
                    torch.randn(100),
                    torch.randn(100),
                    torch.randn(100),
                ],
            },
            # Test case 5: Single explanation layer
            {
                "feature_keys": ("final_layer",),
                "explanations": [
                    SampleExplanation(value=(torch.randn(1, 10),)),
                    SampleExplanation(value=(torch.randn(1, 10),)),
                    SampleExplanation(value=(torch.randn(1, 10),)),
                ],
                "model_outputs": [
                    torch.randn(1, 10),
                    torch.randn(1, 10),
                    torch.randn(1, 10),
                ],
                "frozen_features": [None, None, None],
            },
        ],
    )
    def test_batch_unbatch_roundtrip(
        self,
        sample_ids: list[str],
        sample_targets: list[SampleExplanationTarget],
        explanation_data: dict[str, Any],
    ):
        """Test that batch -> unbatch preserves all explanation state data."""
        # Create individual samples
        samples = []
        for i, sample_id in enumerate(sample_ids):
            sample = SampleExplanationState(
                sample_id=sample_id,
                target=sample_targets[i],
                feature_keys=explanation_data["feature_keys"],
                frozen_features=explanation_data["frozen_features"][i],
                explanations=explanation_data["explanations"][i],
                model_outputs=explanation_data["model_outputs"][i],
            )
            samples.append(sample)

        # Batch the samples
        batched = BatchExplanationState.fromlist(samples)

        # Unbatch back to samples
        unbatched = batched.tolist()

        # Verify we get the same number of samples
        assert len(unbatched) == len(samples)
        assert len(unbatched) == len(sample_ids)

        # Verify each sample matches the original
        for original, recovered in zip(samples, unbatched, strict=True):
            assert original.sample_id == recovered.sample_id
            assert original.feature_keys == recovered.feature_keys
            assert original.is_multitarget == recovered.is_multitarget
            assert original.target == recovered.target

            # Check model outputs
            assert torch.equal(original.model_outputs, recovered.model_outputs)

            # Check frozen features
            if original.frozen_features is not None:
                assert recovered.frozen_features is not None
                assert torch.equal(original.frozen_features, recovered.frozen_features)
            else:
                assert recovered.frozen_features is None

            # Check explanations
            assert len(original.explanations.value) == len(recovered.explanations.value)
            for orig_exp, recv_exp in zip(
                original.explanations.value, recovered.explanations.value, strict=True
            ):
                assert torch.equal(orig_exp, recv_exp), (
                    "Explanation tensors should be identical"
                )

    def test_multitarget_batch_unbatch_roundtrip(self, sample_ids: list[str]):
        """Test multitarget explanation state batching/unbatching."""
        # Create multitarget samples
        samples = []
        for i, sample_id in enumerate(sample_ids):
            targets = [
                SampleExplanationTarget(value=i, name=f"target_{i}"),
                SampleExplanationTarget(value=i + 10, name=f"target_{i}_alt"),
            ]
            explanations = MultiTargetSampleExplanation(
                value=[
                    SampleExplanation(
                        value=(torch.randn(1, 32), torch.randn(1, 64))
                    ),  # explanations for target 1
                    SampleExplanation(
                        value=(torch.randn(1, 32), torch.randn(1, 64))
                    ),  # explanations for target 2
                ]
            )

            sample = SampleExplanationState(
                sample_id=sample_id,
                target=targets,
                feature_keys=("layer1", "layer2"),
                frozen_features=None,
                explanations=explanations,
                model_outputs=torch.randn(1, 5),
            )
            samples.append(sample)

        # Batch the samples
        batched = BatchExplanationState.fromlist(samples)

        # Verify batch properties
        assert batched.is_multitarget is True
        assert isinstance(batched.target, list)
        assert len(batched.target) == 2  # Two targets per sample
        assert isinstance(batched.explanations, MultiTargetBatchExplanation)
        assert batched.explanations.n_targets == 2  # Two explanation sets

        # Unbatch back to samples
        unbatched = batched.tolist()

        # Verify we get the same number of samples
        assert len(unbatched) == len(samples)

        # Verify each sample matches the original
        for original, recovered in zip(samples, unbatched, strict=True):
            assert original.sample_id == recovered.sample_id
            assert original.is_multitarget == recovered.is_multitarget
            assert len(original.target) == len(recovered.target)

            # Check targets
            for orig_tgt, recv_tgt in zip(
                original.target, recovered.target, strict=True
            ):
                assert orig_tgt.value == recv_tgt.value
                assert orig_tgt.name == recv_tgt.name

            # Check explanations
            assert len(original.explanations.value) == len(recovered.explanations.value)
            for orig_exp_set, recv_exp_set in zip(
                original.explanations.value, recovered.explanations.value, strict=True
            ):
                assert len(orig_exp_set.value) == len(recv_exp_set.value)
                for orig_exp, recv_exp in zip(
                    orig_exp_set.value, recv_exp_set.value, strict=True
                ):
                    assert torch.equal(orig_exp, recv_exp)

    def test_empty_batch(self):
        """Test handling of empty batch."""
        with pytest.raises(AssertionError):
            BatchExplanationState.fromlist([])

    def test_single_sample_batch(
        self, sample_ids: list[str], sample_targets: list[SampleExplanationTarget]
    ):
        """Test batching/unbatching with single sample."""
        sample = SampleExplanationState(
            sample_id=sample_ids[0],
            target=sample_targets[0],
            feature_keys=("conv1", "fc"),
            frozen_features=None,
            explanations=SampleExplanation(
                value=(torch.ones(1, 32), torch.zeros(1, 10))
            ),
            model_outputs=torch.randn(1, 5),
        )

        batched = BatchExplanationState.fromlist([sample])
        unbatched = batched.tolist()

        assert len(unbatched) == 1
        assert unbatched[0].sample_id == sample.sample_id
        assert unbatched[0].feature_keys == sample.feature_keys

        for orig, recv in zip(
            sample.explanations.value, unbatched[0].explanations.value, strict=True
        ):
            assert torch.equal(orig, recv)

    def test_batch_properties(
        self, sample_ids: list[str], sample_targets: list[SampleExplanationTarget]
    ):
        """Test that batch properties are correctly set."""
        samples = []
        for i, (sid, target) in enumerate(zip(sample_ids, sample_targets, strict=True)):
            sample = SampleExplanationState(
                sample_id=sid,
                target=target,
                feature_keys=("layer1", "layer2"),
                frozen_features=None,
                explanations=SampleExplanation(
                    value=(torch.randn(1, 16), torch.randn(1, 32))
                ),
                model_outputs=torch.randn(1, 3),
            )
            samples.append(sample)

        batched = BatchExplanationState.fromlist(samples)

        assert batched.sample_id == sample_ids
        assert batched.feature_keys == ("layer1", "layer2")
        assert batched.batch_size == len(sample_ids)
        assert batched.is_multitarget is False

        # Check that each batched explanation has correct batch dimension
        for explanation_tensor in batched.explanations.value:
            assert explanation_tensor.size(0) == len(sample_ids)

        # Check model outputs batch dimension
        assert batched.model_outputs.size(0) == len(sample_ids)

    def test_inconsistent_feature_keys_requirement(
        self, sample_ids: list[str], sample_targets: list[SampleExplanationTarget]
    ):
        """Test that all samples must have the same feature keys."""
        sample1 = SampleExplanationState(
            sample_id=sample_ids[0],
            target=sample_targets[0],
            feature_keys=("layer1", "layer2"),
            frozen_features=None,
            explanations=SampleExplanation(
                value=(torch.randn(1, 16), torch.randn(1, 32))
            ),
            model_outputs=torch.randn(1, 5),
        )
        sample2 = SampleExplanationState(
            sample_id=sample_ids[1],
            target=sample_targets[1],
            feature_keys=("layer1", "layer3"),  # Different key
            frozen_features=None,
            explanations=SampleExplanation(
                value=(torch.randn(1, 16), torch.randn(1, 32))
            ),
            model_outputs=torch.randn(1, 5),
        )

        # This should fail due to inconsistent feature keys
        with pytest.raises((ValueError, AssertionError)):
            BatchExplanationState.fromlist([sample1, sample2])

    def test_inconsistent_multitarget_requirement(
        self, sample_ids: list[str], sample_targets: list[SampleExplanationTarget]
    ):
        """Test that all samples must have the same multitarget setting."""
        sample1 = SampleExplanationState(
            sample_id=sample_ids[0],
            target=sample_targets[0],
            feature_keys=("layer1",),
            frozen_features=None,
            explanations=SampleExplanation(value=(torch.randn(1, 16),)),
            model_outputs=torch.randn(1, 5),
        )
        sample2 = SampleExplanationState(
            sample_id=sample_ids[1],
            target=[sample_targets[1]],  # List format for multitarget
            feature_keys=("layer1",),
            frozen_features=None,
            explanations=MultiTargetSampleExplanation(
                value=[SampleExplanation(value=(torch.randn(1, 16),))]
            ),
            model_outputs=torch.randn(1, 5),
        )

        # This should fail due to inconsistent multitarget settings
        with pytest.raises((ValueError, AssertionError)):
            BatchExplanationState.fromlist([sample1, sample2])

    def test_to_device(self, sample_ids: list[str]):
        """Test device movement functionality."""
        batch_state = BatchExplanationState(
            sample_id=sample_ids,
            target=None,
            feature_keys=("layer1",),
            frozen_features=[torch.randn(10) for _ in sample_ids],
            explanations=BatchExplanation(value=(torch.randn(len(sample_ids), 32),)),
            model_outputs=torch.randn(len(sample_ids), 5),
        )

        # Test device movement (CPU to CPU in this case)
        moved_state = batch_state.to_device("cpu")

        assert moved_state.sample_id == batch_state.sample_id
        assert moved_state.feature_keys == batch_state.feature_keys

        # Check that tensors are on the correct device
        assert moved_state.explanations.device.type == "cpu"

        if moved_state.frozen_features is not None:
            for frozen_feat in moved_state.frozen_features:
                assert frozen_feat.device.type == "cpu"

    def test_none_target_handling(self, sample_ids: list[str]):
        """Test handling of None targets."""
        samples = []
        for sid in sample_ids:
            sample = SampleExplanationState(
                sample_id=sid,
                target=None,
                feature_keys=("layer1",),
                frozen_features=None,
                explanations=SampleExplanation(value=(torch.randn(1, 16),)),
                model_outputs=torch.randn(1, 5),
            )
            samples.append(sample)

        batched = BatchExplanationState.fromlist(samples)
        assert batched.target is None

        unbatched = batched.tolist()
        for sample in unbatched:
            assert sample.target is None
