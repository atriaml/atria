from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from atria_logger import get_logger

if TYPE_CHECKING:
    from atria_prv.mia.configs import AttackConfig

logger = get_logger(__name__)


class _IdentityModel(torch.nn.Module):
    """Surrogate model whose output equals its input.

    ART's black-box attack drives the target model through ``estimator.predict(x)``
    to obtain prediction-probability features. Our token-classification model cannot
    be predicted on that way (it consumes a ``DocumentTensorDataModel`` and emits
    per-token outputs), so we pre-compute each document's mean softmax vector and feed
    it as ``x``. Wrapping this identity module in an ART ``PyTorchClassifier`` makes
    ``predict(p) == p``, so ART sees exactly the per-document probabilities we
    collected. The real target model is never run inside ART.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class MembershipInferenceAttack:
    """Applies a membership inference attack (ART) to collected per-document signals.

    The attack type is taken from the config and applied via ART, treating each
    document as one sample:
      - ``"black_box"``: a trainable attack model (rf / gb / nn), fit on the first
        portion of each set and scored on the held-out remainder, plus ROC-AUC and
        worst-case TPR@FPR (``get_roc_for_fpr``).
      - ``"rule_based"``: the untrained rule-based attack (member iff the model
        classifies the document correctly). Requires no member fitting.

    If ``adversary_has_trainset_access`` is False the attack cannot fit on real
    members, so it falls back to ``"rule_based"``.
    """

    def __init__(self, cfg: AttackConfig) -> None:
        self._cfg = cfg

    # ------------------------------------------------------------------ public
    def run(
        self,
        *,
        num_labels: int,
        x_members: np.ndarray,
        y_members: np.ndarray,
        x_nonmembers: np.ndarray,
        y_nonmembers: np.ndarray,
    ) -> dict[str, Any]:
        estimator = self._identity_estimator(num_labels)

        attack_type = self._cfg.attack_type
        if not self._cfg.adversary_has_trainset_access and attack_type != "rule_based":
            logger.warning(
                "adversary_has_trainset_access=False -> forcing attack_type='rule_based' "
                "(cannot fit an attack model without access to the training set)."
            )
            attack_type = "rule_based"

        x_m = x_members.astype(np.float32)
        y_m = y_members.astype(np.int64)
        x_n = x_nonmembers.astype(np.float32)
        y_n = y_nonmembers.astype(np.int64)

        if attack_type == "rule_based":
            result = self._rule_based(estimator, x_m, y_m, x_n, y_n)
        else:
            result = self._black_box(estimator, x_m, y_m, x_n, y_n)
        result["attack_type"] = attack_type
        logger.info(f"Membership inference attack ({attack_type}): {result}")
        return result

    # ------------------------------------------------------------------ internals
    def _identity_estimator(self, num_labels: int):
        from art.estimators.classification import PyTorchClassifier

        return PyTorchClassifier(
            model=_IdentityModel(),
            loss=torch.nn.CrossEntropyLoss(),
            optimizer=None,
            input_shape=(num_labels,),
            nb_classes=num_labels,
            clip_values=(0.0, 1.0),
        )

    def _rule_based(self, estimator, x_m, y_m, x_n, y_n) -> dict[str, Any]:
        from art.attacks.inference.membership_inference import (
            MembershipInferenceBlackBoxRuleBased,
        )

        attack = MembershipInferenceBlackBoxRuleBased(estimator)
        return _accuracy_report(attack.infer(x_m, y_m), attack.infer(x_n, y_n))

    def _black_box(self, estimator, x_m, y_m, x_n, y_n) -> dict[str, Any]:
        from art.attacks.inference.membership_inference import (
            MembershipInferenceBlackBox,
        )
        from art.metrics.privacy.worst_case_mia_score import get_roc_for_fpr
        from sklearn.metrics import roc_auc_score

        a_m = int(len(x_m) * self._cfg.attack_train_ratio)
        a_n = int(len(x_n) * self._cfg.attack_train_ratio)

        attack = MembershipInferenceBlackBox(
            estimator, attack_model_type=self._cfg.attack_model_type
        )
        attack.fit(x_m[:a_m], y_m[:a_m], x_n[:a_n], y_n[:a_n])

        inferred_members = attack.infer(x_m[a_m:], y_m[a_m:])
        inferred_nonmembers = attack.infer(x_n[a_n:], y_n[a_n:])
        report = _accuracy_report(inferred_members, inferred_nonmembers)

        prob_members = np.squeeze(
            attack.infer(x_m[a_m:], y_m[a_m:], probabilities=True), axis=-1
        )
        prob_nonmembers = np.squeeze(
            attack.infer(x_n[a_n:], y_n[a_n:], probabilities=True), axis=-1
        )
        attack_proba = np.concatenate([prob_members, prob_nonmembers])
        attack_true = np.concatenate(
            [np.ones(len(prob_members)), np.zeros(len(prob_nonmembers))]
        )
        report["auc"] = float(roc_auc_score(attack_true, attack_proba))

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

        if self._cfg.per_class_worst_case:
            target_labels = np.concatenate([y_m[a_m:], y_n[a_n:]])
            report["worst_case_per_class"] = [
                {
                    "class": int(cls),
                    "tpr": float(t),
                    "fpr": float(f),
                    "threshold": float(th),
                }
                for cls, f, t, th in get_roc_for_fpr(
                    attack_proba=attack_proba,
                    attack_true=attack_true,
                    targeted_fpr=self._cfg.targeted_fpr,
                    target_model_labels=target_labels,
                )
            ]
        return report


def _accuracy_report(
    inferred_members: np.ndarray, inferred_nonmembers: np.ndarray
) -> dict[str, Any]:
    """Notebook-style attack accuracy plus precision/recall.

    ``inferred_members`` should ideally be all ``1`` (members flagged as members)
    and ``inferred_nonmembers`` all ``0``.
    """
    from sklearn.metrics import precision_recall_fscore_support

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
