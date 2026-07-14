from __future__ import annotations

import pytest

from atria_prv.att._attacks._attack import (
    MembershipInferenceAttack,
    MembershipInferenceBlackBox,
)
from atria_prv.att.configs import AttackConfig

_EXPECTED_REPORT_KEYS = {
    "member_acc",
    "nonmember_acc",
    "balanced_acc",
    "precision",
    "recall",
    "n_members",
    "n_nonmembers",
    "auc",
    "roc_curve",
    "worst_case",
}


def test_membership_inference_attack_wraps_black_box():
    """MembershipInferenceAttack is our own wrapper, not an ART subclass; it composes
    a MembershipInferenceBlackBox internally (translating the "mlp" config alias)."""
    attack = MembershipInferenceAttack(AttackConfig(attack_model_type="mlp"))
    assert isinstance(attack._attack, MembershipInferenceBlackBox)
    assert attack._attack.attack_model_type == "nn"


def test_black_box_nn_uses_real_feature_width(separable_features):
    """The "nn" attack model's input layer is sized off the real feature width, not a
    caller-supplied class count -- ART's own no-label "nn" path used to size it off the
    (here, meaningless) target-estimator class count instead."""
    attack = MembershipInferenceBlackBox(attack_model_type="nn")
    num_features = separable_features["features_members_train"].shape[1]

    attack.fit(
        members_x=separable_features["features_members_train"].to_numpy(),
        non_members_x=separable_features["features_nonmembers_train"].to_numpy(),
    )
    assert attack.attack_model.num_features == num_features


def test_membership_inference_attack_on_separable_features(separable_features):
    report = MembershipInferenceAttack(AttackConfig(attack_model_type="rf")).run(
        **separable_features
    )

    assert set(report.keys()) == _EXPECTED_REPORT_KEYS
    assert report["auc"] > 0.9
    assert report["worst_case"]["targeted_fpr"] == AttackConfig().targeted_fpr


@pytest.mark.parametrize("attack_model_type", ["rf", "gb", "lr", "nn", "mlp"])
def test_membership_inference_attack_all_model_types(
    separable_features, attack_model_type
):
    """Every attack_model_type runs without needing a class count anywhere.

    "nn" in particular used to be broken in ART's own MembershipInferenceBlackBox for
    feature-vector inputs: with no labels supplied, it sizes its internal attack
    network's input layer off the (here, meaningless) target-estimator class count
    instead of the actual feature width, so it would crash on any real feature vector
    whose width doesn't happen to equal that class count.
    """
    report = MembershipInferenceAttack(
        AttackConfig(attack_model_type=attack_model_type)
    ).run(**separable_features)
    assert report["auc"] > 0.85
