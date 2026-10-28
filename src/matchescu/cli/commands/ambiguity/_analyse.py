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

from ._alpha_sweep import AlphaSweep, AlphaSweepResult
from ._config import ThresholdAnalysisConfig
from ._generate import AmbiguityGenerator
from ._gmm_fit import GmmFit, InsufficientScoresError
from ._plotting import (
    render_alpha_sweep_plot,
    render_gmm_fit_plot,
    render_threshold_plot,
)
from ._shared import load_ambiguity_config, load_benchmark_data
from .selection import AlphaSelector, SelectionSettings

_DATASET_DISPLAY_NAMES: dict[str, str] = {
    "affiliationstrings": "affiliations",
    "cora1": "cora",
    "fodors_zagat_nophone": "fodors-zagat-nophone",
    "geographicalSettelments": "geographic-settlements",
}

DISPLAY_DECIMALS: int = 2


def analyse_threshold(
    threshold: ThresholdAnalysisConfig,
    data: CsvBenchmarkData,
    generator: AmbiguityGenerator,
) -> tuple[DataFrame, GmmFit, list[float], int]:
    """Compute the threshold sweep and the GMM-optimal cutoff for one dataset.

    The mixture is fitted on the pairing scores made without any threshold,
    so the data-driven cutoff does not depend on the configured operating
    point. The sweep is deliberately one-factor-at-a-time: at evenly spaced
    cutoffs it reports how many bridges generation would actually produce
    with ``min_ambiguity`` set to that cutoff, exposing how sensitive bridge
    availability is to the cutoff alone. The same count is taken at the
    fitted cutoff.

    .. note::
        The pairing runs once for the unfiltered scores and once more for
        every sweep cutoff plus the fitted cutoff, so the scoring cost grows
        with the step count. The fit runs right after the first run, so a
        dataset with too few scores fails before the expensive runs.

    :param threshold: the sweep and the mixture fit settings. The sweep has
        exactly ``step_count`` cutoffs spanning
        ``[start_value, start_value + (step_count - 1) * step_size]``;
        ``gmm_restarts`` trades runtime for a more stable decision boundary.
    :param data: dataset whose bridging potential is analysed; supplies the
        cluster count normalising the bridge counts in the sweep frame and
        the display name that reports and plots reuse.
    :param generator: generator built for `data`; its pairing scores are the
        population the mixture is fitted on, and its pairing strategy decides
        the bridge counts. Its own `min_ambiguity` is ignored here.
    :returns: the sweep frame (columns `dataset`, `threshold`, `bridge_count`,
        `cluster_count`; dataset carries the display name), the resolved
        `GmmFit`, the unfiltered pairing scores the fit was made on, and the
        number of bridges generation would produce at the fitted cutoff.
    :raises InsufficientScoresError: when the dataset yields too few pairing
        scores for the mixture fit; no bridge counting happens in that case.
    """
    print(f"analysing {data.name}")
    scores = generator.best_pairing_scores()
    gmm_fit = GmmFit.from_scores(scores, n_init=threshold.gmm_restarts)
    threshold_range = threshold.start_value + threshold.step_size * np.arange(
        threshold.step_count
    )
    *sweep_counts, bridges_at_theta = generator.bridge_counts(
        [*threshold_range.tolist(), gmm_fit.threshold]
    )

    display_name = _DATASET_DISPLAY_NAMES.get(data.name, data.name)
    cluster_count = len(data.compute_clusters())

    sweep_rows = [
        {
            "dataset": display_name,
            "threshold": t,
            "bridge_count": count,
            "cluster_count": cluster_count,
        }
        for t, count in zip(threshold_range, sweep_counts)
    ]

    return (
        pl.DataFrame(data=sweep_rows),
        gmm_fit,
        scores,
        bridges_at_theta,
    )


def build_methodology_rows(
    analysis_frames: list[DataFrame],
    analysis_fits: list[GmmFit],
    analysis_bridges: list[int],
    analysis_sweeps: list[AlphaSweepResult],
    selection_settings: SelectionSettings,
) -> list[dict[Any, Any]]:
    """Summarise every analysed dataset into methodology report rows.

    Each row condenses one dataset's fit into the numbers populating the
    ambiguity-injection table: bridges available at the fitted threshold,
    normalised per cluster, the mixture component statistics, the alpha
    chosen from the sweep with the parameters of that choice, and the bridge
    components at the configured threshold. Display rounding (2 decimals)
    happens here so the CSV is directly usable, while raw precision remains
    available on the underlying `GmmFit` and in the alpha sweep CSV.

    :param analysis_frames: per-dataset sweep frames (the `dataset` and
        `cluster_count` columns are consumed here), index-aligned with the
        other collections.
    :param analysis_fits: per-dataset fits supplying the theta boundary and
        component statistics.
    :param analysis_bridges: per-dataset number of bridges generation would
        produce at the fitted threshold, as returned by `analyse_threshold`.
    :param analysis_sweeps: per-dataset alpha sweeps supplying the chosen
        alpha and the bridge component columns.
    :param selection_settings: parameters of the alpha choice, identical for
        every dataset.
    :returns: one dict per dataset, in collection order; an empty input yields
        an empty list.
    """
    rows: list[dict[Any, Any]] = []
    for df, fit, bridge_count_at_theta, sweep in zip(
        analysis_frames, analysis_fits, analysis_bridges, analysis_sweeps
    ):
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
                **selection_settings.as_row(),
                **_rounded(sweep.summary()),
            }
        )
    return rows


def _rounded(columns: dict[str, Any]) -> dict[str, Any]:
    return {
        name: round(value, DISPLAY_DECIMALS) if isinstance(value, float) else value
        for name, value in columns.items()
    }


def _write_methodology_csv(rows: list[dict[Any, Any]], out_path: Path) -> None:
    df = pl.DataFrame(rows)
    df.write_csv(out_path, include_header=True)


def write_report(
    analysis_frames: list[DataFrame],
    methodology_rows: list[dict[Any, Any]],
    alpha_sweep_rows: list[dict[str, Any]],
    plot_dir: str | PathLike,
) -> None:
    """Write the tabular part of the ambiguity analysis report.

    The report pairs the per-dataset methodology summary with the full
    threshold sweep documenting the structural sensitivity of the theta
    cutoff, and with the full alpha sweep documenting how the chosen alpha
    was found. All artifacts are CSV-only; figures are the plotting
    concern's responsibility and are rendered by the caller, separately.

    :param analysis_frames: per-dataset threshold sweep frames; their
        concatenation is the threshold sweep artifact, so an empty
        collection writes nothing.
    :param methodology_rows: summary rows built by `build_methodology_rows`.
    :param alpha_sweep_rows: profiled alpha rows of every dataset.
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
    pl.DataFrame(alpha_sweep_rows, infer_schema_length=None).write_csv(
        plot_path / "ambiguity_alpha_sweep.csv", include_header=True
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
    "--plot-dir",
    "plot_dir",
    type=click.Path(
        exists=False, file_okay=False, dir_okay=True, writable=True, resolve_path=True
    ),
    default=".",
    required=False,
    help="Directory where plots, the methodology CSV and the sweep CSVs are written",
)
@click.pass_context
def analyse_ambiguity(
    ctx: Context,
    config_file: str | PathLike,
    plot_dir: str | PathLike,
):
    """Analyse the ambiguity potential of configured benchmark datasets.

    Every setting comes from the configuration file; see the package
    README. Two searches run per dataset. The threshold search fits the
    minimum ambiguity θ from the pairing scores, ignoring the configured
    ``minAmbiguity``. The alpha search generates the dataset's ``alphas`` at
    the configured ``minAmbiguity``, profiles every variant and chooses the
    alpha whose content separability lies inside the bounds, by the chosen
    criterion.
    """
    root_dir = get_options(ctx).root_dir
    config = load_ambiguity_config(config_file, root_dir)
    selection_settings = config.analysis.selection.to_selection_settings()
    try:
        alpha_sweep = AlphaSweep(
            AlphaSelector(selection_settings), config.profiling.to_profile_options()
        )
    except ValueError as err:
        raise click.UsageError(str(err)) from err

    analysis_frames: list[DataFrame] = []
    analysis_fits: list[GmmFit] = []
    analysis_scores: list[list[float]] = []
    analysis_bridges: list[int] = []
    analysis_sweeps: list[AlphaSweepResult] = []
    for input_cfg in config.input_data:
        print("processing", input_cfg.dataset.directory)
        data = load_benchmark_data(input_cfg, root_dir)
        generator = AmbiguityGenerator.from_config(input_cfg, data)
        try:
            sweep_df, gmm_fit, scores, bridges_at_theta = analyse_threshold(
                config.analysis.threshold, data, generator
            )
        except InsufficientScoresError as err:
            warnings.warn(f"skipping {data.name}: {err}")
            continue
        analysis_frames.append(sweep_df)
        analysis_fits.append(gmm_fit)
        analysis_scores.append(scores)
        analysis_bridges.append(bridges_at_theta)
        analysis_sweeps.append(
            alpha_sweep.run(generator, data.id_table, sweep_df["dataset"][0])
        )

    if not analysis_frames:
        warnings.warn("no dataset could be analysed; nothing written")
        return

    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)
    for df, fit, scores, sweep in zip(
        analysis_frames, analysis_fits, analysis_scores, analysis_sweeps
    ):
        name = df["dataset"][0]
        render_gmm_fit_plot(scores, fit, plot_path / f"gmm_fit_{name}.png")
        render_alpha_sweep_plot(
            sweep, selection_settings.bounds, plot_path / f"alpha_sweep_{name}.png"
        )

    methodology_rows = build_methodology_rows(
        analysis_frames,
        analysis_fits,
        analysis_bridges,
        analysis_sweeps,
        selection_settings,
    )
    alpha_sweep_rows = [row for sweep in analysis_sweeps for row in sweep.rows]
    write_report(analysis_frames, methodology_rows, alpha_sweep_rows, plot_path)

    render_threshold_plot(
        analysis_frames,
        analysis_fits,
        analysis_scores,
        analysis_bridges,
        plot_path / "threshold_analysis.png",
    )
