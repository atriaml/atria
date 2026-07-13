from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np


def save_loss_distributions(
    loss_members: Sequence[float] | np.ndarray,
    loss_nonmembers: Sequence[float] | np.ndarray,
    output_path: str | Path,
    *,
    title: str = "MIA loss distributions (train)",
    bins: int = 100,
) -> None:
    """Save overlaid member vs. non-member per-document loss curves for debugging.

    The gap between the two curves is the attack signal: members (seen in training)
    should sit at lower loss than non-members. Draws step-histogram outlines so the
    two distributions read as curves.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    members = np.asarray(loss_members, dtype=np.float64).reshape(-1)
    nonmembers = np.asarray(loss_nonmembers, dtype=np.float64).reshape(-1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # shared bin edges over the combined range so the two curves are comparable
    combined = np.concatenate([members, nonmembers])
    lo, hi = float(np.nanmin(combined)), float(np.nanmax(combined))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    edges = np.linspace(lo, hi, bins + 1)

    plt.figure(figsize=(6, 4))
    for data, color, label in (
        (members, "tab:red", f"members (n={members.size})"),
        (nonmembers, "tab:blue", f"non-members (n={nonmembers.size})"),
    ):
        plt.hist(
            data,
            bins=edges,
            density=True,
            histtype="step",
            lw=2,
            color=color,
            label=f"{label}, mean={np.nanmean(data):.3f}",
        )
        plt.hist(data, bins=edges, density=True, color=color, alpha=0.15)
    plt.xlabel("per-document loss")
    plt.ylabel("density")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_roc_curve(
    fpr: Sequence[float],
    tpr: Sequence[float],
    auc: float,
    output_path: str | Path,
) -> None:
    """Save the membership-inference ROC curve to ``output_path`` as a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Membership Inference ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
