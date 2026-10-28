import warnings
from os import PathLike
from pathlib import Path
from typing import Any

import click
import numpy as np
import polars as pl
from click import Context
from matchescu.matching.evaluation.data.benchmark import CsvBenchmarkData
from polars import DataFrame

from matchescu.cli.runtime import get_options

from ._generate import AmbiguityGenerator
from ._gmm_fit import GmmFit, InsufficientScoresError
from ._plotting import render_gmm_fit_plot, render_threshold_plot
from ._shared import (
    load_ambiguity_config,
    load_benchmark_data,
    parse_alphas,
    parse_secondary_features,
)

_DATASET_DISPLAY_NAMES: dict[str, str] = {
    "affiliationstrings": "affiliations",
    "cora1": "cora",
    "fodors_zagat_nophone": "fodors-zagat-nophone",
    "geographicalSettelments": "geographic-settlements",
}


def analyse_threshold(
    analysis_start_value: float,
    analysis_step_count: float,
    analysis_step_size: float,
    data: CsvBenchmarkData,
    generator: AmbiguityGenerator,
    gmm_restarts: int = 50,
) -> tuple[DataFrame, GmmFit, list[float]]:
    """Compute the threshold sweep and the GMM-optimal cutoff for one dataset.

    The sweep is deliberately one-factor-at-a-time: bridge counts are counted
    at evenly spaced cutoffs over the generator's best-pairing scores,
    exposing how sensitive bridge availability is to the cutoff alone. The
    mixture fit then supplies the data-driven cutoff that the sweep is judged
    against, keeping structural analysis (sweep) and statistical analysis
    (Bayes boundary) separable.

    :param analysis_start_value: first cutoff of the sweep.
    :param analysis_step_count: number of cutoffs in the sweep.
    :param analysis_step_size: distance between consecutive cutoffs.
    :param data: dataset whose bridging potential is analysed; supplies the
        cluster count normalising the bridge counts in the sweep frame and
        the display name that reports and plots reuse.
    :param generator: generator built for `data`; its pairing scores are the
        analysed population. Its `min_ambiguity` filter is ignored for these
        scores, per `best_pairing_scores`.
    :param gmm_restarts: random restarts for the mixture fit; more restarts
        trade runtime for a more stable decision boundary.
    :returns: the sweep frame (columns `dataset`, `threshold`, `bridge_count`,
        `cluster_count`; dataset carries the display name), the resolved
        `GmmFit` and the raw best-pairing scores — one score population feeds
        both artifacts directly and callers reuse it for plots and report
        rows without refitting.
    """
    print(f"analysing {data.name}")
    scores = generator.best_pairing_scores()
    threshold_range = np.arange(
        analysis_start_value,
        analysis_step_size * (analysis_step_count + 1),
        analysis_step_size,
    )
    gmm_fit = GmmFit.from_scores(scores, n_init=gmm_restarts)

    display_name = _DATASET_DISPLAY_NAMES.get(data.name, data.name)
    cluster_count = len(data.compute_clusters())

    sweep_rows = [
        {
            "dataset": display_name,
            "threshold": t,
            "bridge_count": sum(1 for s in scores if s >= t),
            "cluster_count": cluster_count,
        }
        for t in threshold_range
    ]

    return (
        pl.DataFrame(data=sweep_rows),
        gmm_fit,
        scores,
    )


def build_methodology_rows(
    analysis_frames: list[DataFrame],
    analysis_fits: list[GmmFit],
    analysis_scores: list[list[float | int]],
) -> list[dict[Any, Any]]:
    """Summarise every analysed dataset into methodology report rows.

    Each row condenses one dataset's fit into the numbers populating the
    ambiguity-injection table: bridges available at the fitted threshold,
    normalised per cluster, plus the component statistics. Display rounding
    (2 decimals) happens here so the CSV is directly usable, while raw
    precision remains available on the underlying `GmmFit`.

    :param analysis_frames: per-dataset sweep frames (the `dataset` and
        `cluster_count` columns are consumed here), index-aligned with the
        other collections.
    :param analysis_fits: per-dataset fits supplying the theta boundary and
        component statistics.
    :param analysis_scores: per-dataset best-pairing scores; their count above
        the threshold is the bridge availability reported per row.
    :returns: one dict per dataset, in collection order; an empty input yields
        an empty list.
    """
    rows: list[dict[Any, Any]] = []
    for df, fit, scores in zip(analysis_frames, analysis_fits, analysis_scores):
        bridge_count_at_theta = fit.bridge_count(scores)
        rows.append(
            {
                "dataset": df["dataset"][0],
                "cluster_count": df["cluster_count"][0],
                "bridge_count_at_theta": int(bridge_count_at_theta),
                "bridge_count_per_cluster": round(
                    bridge_count_at_theta / df["cluster_count"][0], 2
                ),
                "theta": round(fit.threshold, 2),
                "mu_genuine": round(fit.mu_genuine, 2),
                "sigma_genuine": round(fit.sigma_genuine, 2),
                "pi_genuine": round(fit.pi_genuine, 2),
                "mu_superficial": round(fit.mu_superficial, 2),
                "sigma_superficial": round(fit.sigma_superficial, 2),
                "pi_superficial": round(fit.pi_superficial, 2),
                "boundary_type": fit.boundary_type,
            }
        )
    return rows


def _write_methodology_csv(rows: list[dict[Any, Any]], out_path: Path) -> None:
    df = pl.DataFrame(rows)
    df.write_csv(out_path, include_header=True)


def write_report(
    analysis_frames: list[DataFrame],
    methodology_rows: list[dict[Any, Any]],
    plot_dir: str | PathLike,
) -> None:
    """Write the tabular part of the ambiguity analysis report.

    The report pairs the per-dataset methodology summary with the full
    threshold sweep documenting the structural sensitivity of the theta
    cutoff. Both artifacts are CSV-only; figures are the plotting concern's
    responsibility and are rendered by the caller, separately.

    :param analysis_frames: per-dataset sweep frames; their concatenation is
        the sweep artifact, so an empty collection writes nothing.
    :param methodology_rows: summary rows built by `build_methodology_rows`.
    :param plot_dir: report directory; created when missing.
    :returns: None, and nothing is written when `analysis_frames` is empty.
    """
    if not analysis_frames:
        return

    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    _write_methodology_csv(methodology_rows, plot_path / "ambiguity_methodology.csv")

    # The full threshold sweep documents the structural theta-sensitivity for
    # the supplemental data package.
    pl.concat(analysis_frames).write_csv(
        plot_path / "ambiguity_threshold_sweep.csv", include_header=True
    )


@click.command("analyse")
@click.option(
    "-c",
    "--config-file",
    required=True,
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True
    ),
    help="configuration file for the ambiguity analysis",
)
@click.option(
    "--threshold-analysis-step-count",
    "analysis_step_count",
    type=click.FLOAT,
    default=10.0,
    required=False,
    help="How many thresholds to analyse using one-factor-at-a-time",
)
@click.option(
    "--threshold-analysis-step-size",
    "analysis_step_size",
    type=click.FLOAT,
    default=1.0,
    required=False,
    help="How large should the difference between consecutive thresholds be for one-factor-at-a-time analysis",
)
@click.option(
    "--threshold-analysis-start-value",
    "analysis_start_value",
    type=click.FLOAT,
    default=0.0,
    required=False,
    help="First threshold value of the one-factor-at-a-time sweep",
)
@click.option(
    "--gmm-restarts",
    "gmm_restarts",
    type=click.INT,
    default=50,
    required=False,
    help="Number of random restarts for the Gaussian mixture model fit",
)
@click.option(
    "--plot-dir",
    "plot_dir",
    type=click.Path(
        exists=False, file_okay=False, dir_okay=True, writable=True, resolve_path=True
    ),
    default=".",
    required=False,
    help="Directory where plots and the methodology CSV are written",
)
@click.option(
    "--secondary-features",
    "secondary_features",
    default=None,
    help="Comma-separated secondary features to copy from the home representative.",
)
@click.option(
    "--alphas",
    "alphas",
    type=click.STRING,
    default="0.5",
    required=False,
    help="Comma-separated alpha weights (in (0, 1)) for the harmonic-mean pick",
)
@click.option(
    "--target-floor",
    "target_floor",
    type=click.FLOAT,
    default=0.0,
    required=False,
    help="Floor on sim_target, as a fraction of the symmetric-pick sim_target (in [0, 1])",
)
@click.option(
    "--target-links",
    "target_links",
    type=click.INT,
    default=1,
    required=False,
    help="How many links each bridge gets inside its target cluster",
)
@click.pass_context
def analyse_ambiguity(
    ctx: Context,
    config_file: str | PathLike,
    analysis_step_count: float,
    analysis_step_size: float,
    analysis_start_value: float,
    gmm_restarts: int,
    plot_dir: str | PathLike,
    secondary_features: str | None,
    alphas: str,
    target_floor: float,
    target_links: int,
):
    """Analyse the ambiguity potential of configured benchmark datasets."""
    root_dir = get_options(ctx).root_dir
    config = load_ambiguity_config(config_file, root_dir)
    cli_secondary_features = parse_secondary_features(secondary_features)
    alphas_tuple = parse_alphas(alphas)

    analysis_frames: list[DataFrame] = []
    analysis_fits: list[GmmFit] = []
    analysis_scores: list[list[float]] = []
    for input_cfg in config.input_data:
        print("processing", input_cfg.dataset.directory)
        data = load_benchmark_data(input_cfg, root_dir)
        generator = AmbiguityGenerator.from_config(
            input_cfg,
            data,
            secondary_features=cli_secondary_features,
            alphas=alphas_tuple,
            target_floor=target_floor,
            target_links=target_links,
        )
        try:
            sweep_df, gmm_fit, scores = analyse_threshold(
                analysis_start_value,
                analysis_step_count,
                analysis_step_size,
                data,
                generator,
                gmm_restarts=gmm_restarts,
            )
        except InsufficientScoresError as err:
            warnings.warn(f"skipping {data.name}: {err}")
            continue
        analysis_frames.append(sweep_df)
        analysis_fits.append(gmm_fit)
        analysis_scores.append(scores)

    if not analysis_frames:
        warnings.warn("no dataset could be analysed; nothing written")
        return

    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)
    for df, fit, scores in zip(analysis_frames, analysis_fits, analysis_scores):
        bell_path = plot_path / f"gmm_fit_{df['dataset'][0]}.png"
        render_gmm_fit_plot(scores, fit, bell_path)

    methodology_rows = build_methodology_rows(
        analysis_frames, analysis_fits, analysis_scores
    )
    write_report(analysis_frames, methodology_rows, plot_path)

    render_threshold_plot(
        analysis_frames,
        analysis_fits,
        analysis_scores,
        plot_path / "threshold_analysis.png",
    )
