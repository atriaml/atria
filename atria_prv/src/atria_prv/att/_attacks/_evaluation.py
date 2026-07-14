from __future__ import annotations

from typing import Any

import numpy as np


def _accuracy_report(
    inferred_members: np.ndarray, inferred_nonmembers: np.ndarray
) -> dict[str, Any]:
    """Attack accuracy plus precision/recall.

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


def _tpr_at_targeted_fpr(
    fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray, targeted_fpr: float
) -> tuple[float, float, float]:
    """Snap to the last ROC point with fpr <= targeted_fpr (no interpolation)."""
    idx = np.searchsorted(fpr, targeted_fpr, side="right") - 1
    return float(fpr[idx]), float(tpr[idx]), float(thresholds[idx])
