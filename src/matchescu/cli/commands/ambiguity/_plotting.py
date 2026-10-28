from os import PathLike

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from ._gmm_fit import GmmFit


def render_threshold_plot(
    all_frames: list[pl.DataFrame],
    all_fits: list[GmmFit],
    all_scores: list[list[float]],
    out_path: str | PathLike,
) -> None:
    """Render the threshold-vs-bridge-count sweep as one panel per dataset.

    The 1×N faceted grid lets every panel carry its own scales, with the
    x-axis cropped to the action region `[min(scores) - 1, max(scores) + 1]`,
    so panels compare curve shapes rather than absolute positions. A vertical
    line per panel marks the GMM threshold, annotated with the threshold value
    and the bridge count it selects.

    This is a supplemental artifact: dataset name and boundary type travel on
    the panel titles rather than in any exported data.

    :param all_frames: per-dataset sweep frames, index-aligned with the other
        two collections.
    :param all_fits: per-dataset fits supplying the marked thresholds.
    :param all_scores: per-dataset best-pairing scores defining the action
        region and the annotated bridge counts.
    :param out_path: destination image path; the figure is written as PNG at
        300 dpi and closed before returning.
    """
    sns.set_theme(style="white", font_scale=1.0)
    n = len(all_frames)
    _fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, df, fit, scores in zip(axes, all_frames, all_fits, all_scores):
        dataset_name = df["dataset"][0]
        thresholds = df["threshold"].to_list()
        counts = df["bridge_count"].to_list()

        ax.plot(thresholds, counts, color="#1f77b4", linewidth=1.8)
        ax.axvline(x=fit.threshold, color="black", linestyle="--", linewidth=1.0)

        # Crop x-axis to the action region.
        scores_arr = np.array(scores, dtype=float)
        x_lo = float(scores_arr.min()) - 1.0
        x_hi = float(scores_arr.max()) + 1.0
        ax.set_xlim(x_lo, x_hi)

        bridge_at_theta = fit.bridge_count(scores)
        ann = f"θ={fit.threshold:.2f}\nb={bridge_at_theta}"
        ax.annotate(
            ann,
            xy=(fit.threshold, counts[0] if counts else 0),
            xytext=(0.5, 0.92),
            textcoords="axes fraction",
            ha="center",
            fontsize=7,
            color="black",
        )

        ax.set_title(f"{dataset_name} ({fit.boundary_type})", fontsize=9)
        ax.set_xlabel("θ", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("Bridge count", fontsize=8)
        ax.tick_params(labelsize=7)
        sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def render_gmm_fit_plot(
    scores: list[float], fit: GmmFit, out_path: str | PathLike
) -> None:
    """Render the best-pairing-score histogram with the fitted mixture densities.

    The histogram of scores is overlaid with the two weighted normal
    components (genuine vs superficial overlap) and the resolved threshold,
    giving a visual check that the Bayes boundary separates the populations.
    The x range extends symmetrically beyond the data to cover three standard
    deviations of both components, so the density tails stay visible.

    :param scores: best-pairing scores sampled per dataset.
    :param fit: the fit whose component densities and `threshold` are drawn.
    :param out_path: destination image path; the figure is written as PNG at
        300 dpi and closed before returning.
    """
    sns.set_theme(style="white", font_scale=1.1)
    _fig, ax = plt.subplots(figsize=(7, 4))

    scores_arr = np.array(scores, dtype=float)
    lo = float(min(scores_arr.min(), fit.mu_superficial - 3 * fit.sigma_superficial))
    hi = float(max(scores_arr.max(), fit.mu_genuine + 3 * fit.sigma_genuine))
    x = np.linspace(lo, hi, 400)

    ax.hist(
        scores_arr,
        bins="auto",
        density=True,
        color="lightgrey",
        edgecolor="white",
        label="best-pairing scores",
    )

    genuine_density = fit.genuine_density(x)
    superficial_density = fit.superficial_density(x)

    ax.plot(x, genuine_density, color="#2ca02c", linewidth=2, label="genuine overlap")
    ax.plot(
        x,
        superficial_density,
        color="#d62728",
        linewidth=2,
        label="superficial overlap",
    )
    ax.axvline(
        x=fit.threshold,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"θ={fit.threshold:.2f}",
    )

    ax.set_xlabel("Ambiguity score")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, frameon=True)
    sns.despine()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
