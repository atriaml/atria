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


def save_feature_distributions(
    members_df,
    nonmembers_df,
    output_path: str | Path,
    *,
    feature_columns: Sequence[str] | None = None,
    title: str = "MIA feature distributions (train)",
    bins: int = 100,
    ncols: int = 4,
    max_per_image: int = 24,
) -> list[Path]:
    """Save member vs. non-member distributions for every feature as a subplot grid.

    One subplot per feature column: overlaid step-histograms of members (red) vs.
    non-members (blue), each on its own shared bin range so the two curves compare.
    If there are more than ``max_per_image`` features, they are split across several
    images (``<stem>_1.png``, ``<stem>_2.png``, ...). Returns the list of paths written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = (
        list(feature_columns)
        if feature_columns is not None
        else [c for c in members_df.columns if c in nonmembers_df.columns]
    )
    columns = [c for c in columns if "mean" in c]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # split columns into pages of at most max_per_image features
    pages = [
        columns[i : i + max_per_image] for i in range(0, len(columns), max_per_image)
    ] or [[]]
    multi_page = len(pages) > 1

    written: list[Path] = []
    for page_idx, page_cols in enumerate(pages, start=1):
        if multi_page:
            page_path = output_path.with_name(
                f"{output_path.stem}_{page_idx}{output_path.suffix}"
            )
            page_title = f"{title} ({page_idx}/{len(pages)})"
        else:
            page_path = output_path
            page_title = title

        n = len(page_cols)
        page_ncols = min(ncols, n) if n else 1
        nrows = int(np.ceil(n / page_ncols)) if n else 1
        fig, axes = plt.subplots(
            nrows, page_ncols, figsize=(4 * page_ncols, 3 * nrows), squeeze=False
        )

        for idx, col in enumerate(page_cols):
            ax = axes[idx // page_ncols][idx % page_ncols]
            members = np.asarray(members_df[col].to_numpy(), dtype=np.float64).reshape(
                -1
            )
            nonmembers = np.asarray(
                nonmembers_df[col].to_numpy(), dtype=np.float64
            ).reshape(-1)

            combined = np.concatenate([members, nonmembers])
            combined = combined[np.isfinite(combined)]
            if combined.size:
                lo, hi = float(np.nanmin(combined)), float(np.nanmax(combined))
            else:
                lo, hi = 0.0, 1.0
            if lo == hi:
                lo, hi = lo - 0.5, hi + 0.5
            edges = np.linspace(lo, hi, bins + 1)

            for data, color, label in (
                (members, "tab:red", f"members (n={members.size})"),
                (nonmembers, "tab:blue", f"non-members (n={nonmembers.size})"),
            ):
                ax.hist(
                    data,
                    bins=edges,
                    density=True,
                    histtype="step",
                    lw=2,
                    color=color,
                    label=f"{label}, μ={np.nanmean(data):.3f}",
                )
                ax.hist(data, bins=edges, density=True, color=color, alpha=0.15)
            ax.set_title(col, fontsize=9)
            ax.legend(loc="upper right", fontsize=6)

        # blank out unused axes in the grid
        for idx in range(n, nrows * page_ncols):
            axes[idx // page_ncols][idx % page_ncols].axis("off")

        fig.suptitle(page_title)
        fig.tight_layout()
        fig.savefig(page_path, dpi=150)
        plt.close(fig)
        written.append(page_path)

    return written


def save_roc_curve(
    fpr: Sequence[float],
    tpr: Sequence[float],
    auc: float,
    output_path: str | Path,
    *,
    min_fpr: float = 1e-4,
) -> None:
    """Save the membership-inference ROC curve, both linear and log-log (Carlini et al.).

    No clamping or transformation of values -- fpr/tpr are plotted exactly as given.
    ``min_fpr`` only sets the visible axis window on the log-log plot; points below
    that window are simply outside the frame (matplotlib crops them, same as zooming
    a linear plot), not moved or rewritten.
    """

    import matplotlib.pyplot as plt

    print("HERE")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stem, suffix = output_path.stem, output_path.suffix or ".png"

    fpr = np.asarray(fpr, dtype=np.float64)
    tpr = np.asarray(tpr, dtype=np.float64)

    ref_fprs = [1e-3, 1e-2, 1e-1]
    ref_lines = []
    for target in ref_fprs:
        mask = fpr <= target
        if mask.any():
            ref_lines.append(f"TPR@FPR={target:g}: {tpr[mask].max():.3f}")
    ref_suffix = ("\n" + "  ".join(ref_lines)) if ref_lines else ""

    # --- linear-scale ROC ---
    plt.figure(figsize=(5, 5))
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=1.5,
        marker="o",
        ms=2,
        label=f"ROC (AUC = {auc:.3f})",
    )
    print("HERE")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Membership Inference ROC (linear)" + ref_suffix, fontsize=9)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path.parent / f"{stem}_linear{suffix}", dpi=150)
    plt.close()

    # --- log-log ROC ---
    plt.figure(figsize=(5, 5))
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=1.5,
        marker="o",
        ms=2,
        label=f"ROC (AUC = {auc:.3f})",
    )
    plt.plot(
        [min_fpr, 1], [min_fpr, 1], color="navy", lw=1, linestyle="--", label="chance"
    )
    print("HERE")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim([min_fpr, 1.0])
    plt.ylim([min_fpr, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Membership Inference ROC (log-log)" + ref_suffix, fontsize=9)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path.parent / f"{stem}_loglog{suffix}", dpi=150)
    plt.close()
