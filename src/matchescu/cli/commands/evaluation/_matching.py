import os
from dataclasses import replace
from pathlib import Path

import click
import polars as pl
from matchescu.matching import Matcher
from matchescu.matching.evaluation.data.benchmark import BenchmarkData
from matchescu.reference_store.comparison_space import BinaryComparisonSpace
from matchescu.similarity import GmlGraphPersistence, MatchResult, ReferenceGraph
from matchescu.typing import EntityReference
from pyresolvemetrics import f1, precision, recall
from rich.progress import Progress
from sklearn.metrics import matthews_corrcoef

from matchescu.cli.config import (
    AnyMatcherConfig,
    EvaluationBenchmarkConfig,
    EvaluationConfig,
    new_benchmark_data_factory,
)
from matchescu.cli.data import count_comparison_space_rows, load_comparison_space
from matchescu.cli.models import new_matcher
from matchescu.cli.runtime import get_options, make_absolute_path

from ._eval_opts import EvalOptions
from ._input_order import InputOrder, validate_input_order_option
from ._parallel import UnitReporter, UnitRunner, workers_option
from ._plots import (
    PlotDir,
    plot_dpi_option,
    plot_ground_truth_graph,
    plot_predicted_graph,
)

RESULT_SCHEMA_OVERRIDES = {
    "dataset": pl.String,
    "model": pl.String,
    "left_id": pl.Int32,
    "right_id": pl.Int32,
    "left_source": pl.String,
    "right_source": pl.String,
    "true_label": pl.Int32,
    "normal": pl.Int32,
    "reverse": pl.Int32,
}


def _compute_binary_classifier_metrics(
    true_matches: set, graph: ReferenceGraph, input_order: str
) -> dict:
    # ensure we don't filter out any matches
    actual = set(graph.matches(min_weight=0.0))
    return {
        "input_order": input_order,
        "precision": precision(true_matches, actual),
        "recall": recall(true_matches, actual),
        "f1": f1(true_matches, actual),
    }


def _compute_multiclass_metrics(cs_true_matches, g, order):
    y_true = list(cs_true_matches.values())
    y_pred = []
    for a, b in cs_true_matches:
        if g.has_edge(a, b) and g.has_edge(b, a):
            y_pred.append(1)
        elif g.has_edge(a, b):
            y_pred.append(2)
        elif g.has_edge(b, a):
            y_pred.append(3)
        else:
            y_pred.append(0)
    return {"input_order": order, "mcc": matthews_corrcoef(y_true, y_pred)}


def _uses_binary_metrics(
    model_config: AnyMatcherConfig, degrade_directional: bool
) -> bool:
    return model_config.type != "multiclass" or degrade_directional


class ReferenceGraphFiller:
    def __init__(
        self, input_order: InputOrder, degrade_directional: bool = False
    ) -> None:
        self._degrade_directional = degrade_directional
        self._input_order = input_order

    def __call__(
        self,
        g: ReferenceGraph,
        m: Matcher,
        x: EntityReference,
        y: EntityReference,
        record: dict | None,
    ) -> dict:
        match self._input_order:
            case InputOrder.normal:
                return self._add_normal(g, m, x, y)
            case InputOrder.reverse:
                return self._add_normal(g, m, x, y)
            case InputOrder.both:
                return self._add_both(g, m, x, y)

    def _add_normal(
        self,
        g: ReferenceGraph,
        m: Matcher,
        x: EntityReference,
        y: EntityReference,
        record: dict | None = None,
    ) -> dict:
        match_result = m(x, y)
        g.add(self._optionally_degrade(match_result))
        return self._put_record_attr(record, "normal", match_result.label)

    def _add_reverse(
        self,
        g: ReferenceGraph,
        m: Matcher,
        x: EntityReference,
        y: EntityReference,
        record: dict | None = None,
    ) -> dict:
        match_result = m(y, x)
        g.add(self._optionally_degrade(match_result))
        return self._put_record_attr(record, "reverse", match_result.label)

    def _add_both(
        self,
        g: ReferenceGraph,
        m: Matcher,
        x: EntityReference,
        y: EntityReference,
        record: dict | None = None,
    ) -> dict:
        record = self._add_normal(g, m, x, y, record)
        return self._add_reverse(g, m, x, y, record)

    @staticmethod
    def _put_record_attr(record: dict | None, attr: str, value: object) -> dict:
        r = record or {}
        r[attr] = value
        return r

    def _optionally_degrade(self, result: MatchResult) -> MatchResult:
        if not self._degrade_directional:
            return result
        if result.label == 0 or result.label == 1:
            return result
        return replace(result, label=1)


def _load_dataset(
    ds_config: EvaluationBenchmarkConfig, root_dir: Path, seed: int
) -> tuple[BenchmarkData, BinaryComparisonSpace]:
    builder = new_benchmark_data_factory(ds_config.dataset, root_dir)
    benchmark_data = builder.load_data().create()
    cs = load_comparison_space(
        benchmark_data,
        root_dir / ds_config.dataset.directory,
        ds_config.comparison_space,
        seed=seed,
    )
    return benchmark_data, cs


def _comparison_labels(
    benchmark_data: BenchmarkData, cs: BinaryComparisonSpace
) -> dict[tuple, int]:
    return {cmp: benchmark_data.true_matches.get(cmp, 0) for cmp in cs}


def _plot_ground_truth_unit(
    ds_config: EvaluationBenchmarkConfig,
    seed: int,
    root_dir: Path,
    plot_root: Path,
    dpi: int,
    reporter: UnitReporter,
) -> str:
    """Render one dataset's ground-truth graph and save the shared layout."""
    benchmark_data, cs = _load_dataset(ds_config, root_dir, seed)
    labels = _comparison_labels(benchmark_data, cs)
    plot_dir = PlotDir(plot_root, benchmark_data.name)
    plot_ground_truth_graph(plot_dir, labels, seed, dpi)
    return benchmark_data.name


def _eval_matching_unit(
    ds_config: EvaluationBenchmarkConfig,
    model_config: AnyMatcherConfig,
    input_orders: list[str],
    seed: int,
    root_dir: Path,
    output_path: Path,
    degrade_labels: bool,
    plot_root: Path | None,
    dpi: int,
    reporter: UnitReporter,
) -> list[dict]:
    """Evaluate one (dataset, model) unit: build graphs, save them, score them."""
    try:
    benchmark_data, cs = _load_dataset(ds_config, root_dir, seed)
    cs_pair_gt = _comparison_labels(benchmark_data, cs)
    cs_refs = list(map(benchmark_data.id_table.get_all, cs))

    matcher = new_matcher(model_config, root_dir, ds_config.dataset_alias or benchmark_data.name)
    graphs = {order: ReferenceGraph(directed=True) for order in input_orders}
    model_results = {order: [] for order in input_orders}
    for x, y in cs_refs:
        r = {
            "dataset": benchmark_data.name,
            "model": model_config.name,
            "left_id": x.id.label,
            "left_source": x.id.source,
            "right_id": y.id.label,
            "right_source": y.id.source,
            "true_label": benchmark_data.true_matches.get((x.id, y.id), 0),
        }
        for order, g in graphs.items():
            graph_filler = ReferenceGraphFiller(InputOrder(order), degrade_labels)
            model_results[order].append(graph_filler(g, matcher, x, y, r))
        reporter.advance(len(input_orders))

    graph_dir = output_path / benchmark_data.name / model_config.name
    graph_dir.mkdir(parents=True, exist_ok=True)
    stats = []
    for order, g in graphs.items():
        g.save(GmlGraphPersistence(graph_dir / f"{order}-graph.gml"))

        df = (
            pl.DataFrame(
                data=model_results[order],
                schema_overrides=RESULT_SCHEMA_OVERRIDES,
            )
            .fill_nan(-1)
            .fill_null(-1)
        )
        results_csv_path = graph_dir / f"{order}-result.csv"
        df.write_csv(results_csv_path, include_header=True)

        if _uses_binary_metrics(model_config, degrade_labels):
            true_matches = {cmp for cmp, label in cs_pair_gt.items() if label > 0}
            metrics = _compute_binary_classifier_metrics(true_matches, g, order)
        else:
            metrics = _compute_multiclass_metrics(cs_pair_gt, g, order)
        stats.append(
            {
                "dataset": benchmark_data.name,
                "model": model_config.name,
                **metrics,
            }
        )
        if plot_root is not None:
            plot_predicted_graph(
                PlotDir(plot_root, benchmark_data.name),
                g,
                cs_pair_gt,
                model_config.name,
                order,
                {k: v for k, v in metrics.items() if k != "input_order"},
                ordered=model_config.type == "multiclass",
                dpi=dpi,
            )
    return stats


def _plot_ground_truths(
    runner: UnitRunner,
    cfg: EvaluationConfig,
    root_dir: Path,
    plot_root: Path,
    dpi: int,
) -> None:
    units = [
        (
            f"ground-truth/{ds_config.dataset.directory}",
            (ds_config, cfg.seed, root_dir, plot_root, dpi),
            None,
        )
        for ds_config in cfg.benchmark_data
    ]
    for _ in runner.run(units, _plot_ground_truth_unit):
        pass


@click.command("matching")
@click.option(
    "-R",
    "--reference-graph-dir",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, writable=True, resolve_path=True),
    help="directory where reference graphs will be exported in GML format",
)
@click.option(
    "-s",
    "--stats-csv-path",
    required=True,
    type=click.Path(dir_okay=False, writable=True, resolve_path=True),
    help="CSV file where the matching metrics will be written",
)
@click.option(
    "--graph-plots",
    "graph_plots",
    default=None,
    type=click.Path(file_okay=False, dir_okay=True, writable=True, resolve_path=True),
    help="directory for reference graph plots, one folder per dataset; "
    "plots are only generated when given",
)
@plot_dpi_option()
@click.option(
    "-i",
    "--input-order",
    "input_orders",
    required=False,
    type=click.Choice(InputOrder),
    multiple=True,
    default=[InputOrder.normal],
    callback=validate_input_order_option,
    help="input order to evaluate the matcher in (specify up to 3)",
)
@click.option(
    "--degrade-directional-labels",
    is_flag=True,
    default=False,
    help=(
        "Treat directional matches (label 2) as full matches (label 1) when "
        "building reference graphs, and evaluate multiclass matchers with "
        "binary F1 metrics."
    ),
)
@workers_option()
@click.pass_context
def evaluate_matching(
    ctx: click.Context,
    reference_graph_dir: str | os.PathLike,
    stats_csv_path: str | os.PathLike,
    graph_plots: str | os.PathLike | None,
    plot_dpi: int,
    input_orders: list[str],
    degrade_directional_labels: bool,
    workers: int | None,
):
    """Evaluate a matcher's quality in controlled settings."""
    eval_opts: EvalOptions[EvaluationConfig] = get_options(ctx)
    root_dir = eval_opts.root_dir
    cfg = eval_opts.config
    output_path = make_absolute_path(reference_graph_dir, root_dir)
    csv_path = make_absolute_path(stats_csv_path, root_dir)
    plot_root = make_absolute_path(graph_plots, root_dir) if graph_plots else None

    n_orders = len(input_orders)
    per_ds_count = {
        ds_config.dataset.directory: count_comparison_space_rows(
            root_dir / ds_config.dataset.directory, ds_config.comparison_space
        )
        for ds_config in cfg.benchmark_data
    }

    def _unit_total(ds_dir: str) -> int | None:
        count = per_ds_count[ds_dir]
        return count * n_orders if count is not None else None

    units = [
        (
            f"{ds_config.dataset.directory}/{model_config.name}",
            (
                ds_config,
                model_config,
                list(input_orders),
                cfg.seed,
                root_dir,
                output_path,
                degrade_directional_labels,
                plot_root,
                plot_dpi,
            ),
            _unit_total(ds_config.dataset.directory),
        )
        for ds_config in cfg.benchmark_data
        for model_config in cfg.matching
    ]
    stats = []
    with Progress() as progress:
        runner = UnitRunner(workers, progress)
        if plot_root is not None:
            _plot_ground_truths(runner, cfg, root_dir, plot_root, plot_dpi)
        for _, unit_stats in runner.run(units, _eval_matching_unit):
            stats.extend(unit_stats)
    stats_df = pl.DataFrame(stats)
    stats_df.write_csv(csv_path)
