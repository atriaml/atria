from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from atria_logger import get_logger

if TYPE_CHECKING:
    import pandas as pd

    from atria_prv.att.configs import AttackConfig

logger = get_logger(__name__)


class _IdentityModel(torch.nn.Module):
    """Surrogate model required only to construct the ART estimator.

    We always pass the extracted feature vector to ART via ``pred=`` (and ``y=None``),
    so ART never calls the estimator's ``predict`` / ``compute_loss``; this module just
    satisfies the ``PyTorchClassifier`` constructor.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class MembershipInferenceAttack:
    """Feature-based membership inference attack via ART's black-box attack model.

    Each document is reduced to a named feature vector (see ``TokenSignalExtractor``).
    That vector is fed to ART's ``MembershipInferenceBlackBox`` as the "prediction"
    (``input_type="prediction"``, ``pred=features``, ``y=None`` so no labels are
    needed). ART's attack model (rf / gb / nn / lr) is fit on the attack-train halves
    and scored on the held-out attack-test halves.
    """

    def __init__(self, cfg: AttackConfig) -> None:
        self._cfg = cfg

    def run(
        self,
        *,
        num_labels: int,
        features_members_train: pd.DataFrame,
        features_nonmembers_train: pd.DataFrame,
        features_members_test: pd.DataFrame,
        features_nonmembers_test: pd.DataFrame,
        feature_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        from art.attacks.inference.membership_inference import (
            MembershipInferenceBlackBox,
        )
        from art.metrics.privacy.worst_case_mia_score import get_roc_for_fpr
        from sklearn.metrics import roc_auc_score, roc_curve

        def to_matrix(df: pd.DataFrame) -> np.ndarray:
            columns = feature_columns or list(df.columns)
            return df[columns].to_numpy(dtype=np.float32)

        num_features = len(feature_columns or features_members_train.columns)
        estimator = self._identity_estimator(
            num_labels, num_features
        )  # never called; pred supplied
        attack = MembershipInferenceBlackBox(
            estimator,
            input_type="prediction",
            attack_model_type=self._cfg.attack_model_type,
        )

        # fit on the attack-train halves; features are the only input, y=None (no labels)

        attack.fit(
            x=None,
            y=None,
            test_x=None,
            test_y=None,
            pred=to_matrix(features_members_train),
            test_pred=to_matrix(features_nonmembers_train),
        )

        # score the held-out attack-test halves
        inferred_members = attack.infer(
            None, pred=to_matrix(features_members_test)
        )  # want 1s
        inferred_nonmembers = attack.infer(
            None, pred=to_matrix(features_nonmembers_test)
        )  # want 0s
        report = _accuracy_report(inferred_members, inferred_nonmembers)

        # AUC + worst-case TPR@FPR on the attack-test halves
        prob_members = np.squeeze(
            attack.infer(
                None, pred=to_matrix(features_members_test), probabilities=True
            ),
            axis=-1,
        )
        prob_nonmembers = np.squeeze(
            attack.infer(
                None, pred=to_matrix(features_nonmembers_test), probabilities=True
            ),
            axis=-1,
        )
        attack_proba = np.concatenate([prob_members, prob_nonmembers])
        attack_true = np.concatenate(
            [np.ones(len(prob_members)), np.zeros(len(prob_nonmembers))]
        )
        report["auc"] = float(roc_auc_score(attack_true, attack_proba))

        roc_fpr, roc_tpr, _ = roc_curve(attack_true, attack_proba)
        report["roc_curve"] = {"fpr": roc_fpr.tolist(), "tpr": roc_tpr.tolist()}

        fpr, tpr, threshold = get_roc_for_fpr(
            attack_proba=attack_proba,
            attack_true=attack_true,
            targeted_fpr=self._cfg.targeted_fpr,
        )[0]
        report["worst_case"] = {
            "targeted_fpr": self._cfg.targeted_fpr,
            "tpr": float(tpr),
            "fpr": float(fpr),
            "threshold": float(threshold),
        }

        logger.info(
            f"Membership inference attack ({self._cfg.attack_model_type}): {report}"
        )
        return report

    def _identity_estimator(self, num_labels: int, num_features: int):
        from art.estimators.classification import PyTorchClassifier

        return PyTorchClassifier(
            model=_IdentityModel(),
            loss=torch.nn.CrossEntropyLoss(),
            optimizer=None,
            input_shape=(num_features,),
            nb_classes=num_labels,
        )


def _accuracy_report(
    inferred_members: np.ndarray, inferred_nonmembers: np.ndarray
) -> dict[str, Any]:
    """Attack accuracy plus precision/recall.

    ``inferred_members`` should ideally be all ``1`` (members flagged as members)
    and ``inferred_nonmembers`` all ``0``.
    """
    from sklearn.metrics import precision_recall_fscore_support

    print("inferred_members", inferred_members)
    print("inferred_nonmembers", inferred_nonmembers)

    inferred_members = np.asarray(inferred_members).reshape(-1)
    inferred_nonmembers = np.asarray(inferred_nonmembers).reshape(-1)

    member_acc = float(inferred_members.sum() / len(inferred_members))
    nonmember_acc = float(1 - inferred_nonmembers.sum() / len(inferred_nonmembers))
    balanced_acc = float(
        (member_acc * len(inferred_members) + nonmember_acc * len(inferred_nonmembers))
        / (len(inferred_members) + len(inferred_nonmembers))
    )

    y_pred = np.concatenate([inferred_members, inferred_nonmembers])
    y_true = np.concatenate(
        [np.ones_like(inferred_members), np.zeros_like(inferred_nonmembers)]
    )
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    return {
        "member_acc": member_acc,
        "nonmember_acc": nonmember_acc,
        "balanced_acc": balanced_acc,
        "precision": float(precision),
        "recall": float(recall),
        "n_members": int(len(inferred_members)),
        "n_nonmembers": int(len(inferred_nonmembers)),
    }
