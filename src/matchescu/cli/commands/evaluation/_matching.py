import os
from dataclasses import replace
from functools import partial

import click
import polars as pl
from matchescu.matching import Matcher
from matchescu.similarity import GmlGraphPersistence, MatchResult, ReferenceGraph
from matchescu.typing import EntityReference
from pyresolvemetrics import f1, precision, recall
from rich.progress import Progress
from sklearn.metrics import matthews_corrcoef

from matchescu.cli.config import (
    AnyMatcherConfig,
    EvaluationConfig,
    new_benchmark_data_factory,
)
from matchescu.cli.config._config import EvaluationBenchmarkConfig
from matchescu.cli.data import load_comparison_space
from matchescu.cli.models import new_matcher
from matchescu.cli.runtime import get_options, make_absolute_path

from ._eval_opts import EvalOptions
from ._input_order import InputOrder, validate_input_order_option

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


def _put_record_attr(record: dict, attr: str, value: object) -> dict:
    r = record or {}
    r[attr] = value
    return r


def _degrade_directional_label(result: MatchResult) -> MatchResult:
    if result.label != 2:
        return result
    return replace(result, label=1)


def _matcher_dataset_name(
    ds_config: EvaluationBenchmarkConfig, benchmark_name: str
) -> str:
    return ds_config.dataset_alias or benchmark_name


def _uses_binary_metrics(
    model_config: AnyMatcherConfig, degrade_directional: bool
) -> bool:
    return model_config.type != "multiclass" or degrade_directional


def _add_normal(
    g: ReferenceGraph,
    matcher: Matcher,
    x: EntityReference,
    y: EntityReference,
    record: dict | None = None,
    degrade_directional: bool = False,
) -> dict:
    match_result = matcher(x, y)
    graph_result = (
        _degrade_directional_label(match_result)
        if degrade_directional
        else match_result
    )
    g.add(graph_result)
    return _put_record_attr(record, "normal", match_result.label)


def _add_reverse(
    g: ReferenceGraph,
    matcher: Matcher,
    x: EntityReference,
    y: EntityReference,
    record: dict | None = None,
    degrade_directional: bool = False,
) -> dict:
    match_result = matcher(y, x)
    graph_result = (
        _degrade_directional_label(match_result)
        if degrade_directional
        else match_result
    )
    g.add(graph_result)
    return _put_record_attr(record, "reverse", match_result.label)


def _add_both(
    g: ReferenceGraph,
    matcher: Matcher,
    x: EntityReference,
    y: EntityReference,
    record: dict | None = None,
    degrade_directional: bool = False,
) -> dict:
    record = _add_normal(
        g,
        matcher,
        x,
        y,
        record,
        degrade_directional=degrade_directional,
    )
    return _add_reverse(
        g,
        matcher,
        x,
        y,
        record,
        degrade_directional=degrade_directional,
    )


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
    help="directory where reference graphs will be exported in GML format",
)
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
@click.pass_context
def evaluate_matching(
    ctx: click.Context,
    reference_graph_dir: str | os.PathLike,
    stats_csv_path: str | os.PathLike,
    input_orders: list[str],
    degrade_directional_labels: bool,
):
    """Evaluate a matcher's quality in controlled settings."""
    eval_opts: EvalOptions[EvaluationConfig] = get_options(ctx)
    root_dir = eval_opts.root_dir
    cfg = eval_opts.config
    output_path = make_absolute_path(reference_graph_dir, root_dir)
    csv_path = make_absolute_path(stats_csv_path, root_dir)
    input_order_map = {
        InputOrder.normal: partial(
            _add_normal, degrade_directional=degrade_directional_labels
        ),
        InputOrder.reverse: partial(
            _add_reverse, degrade_directional=degrade_directional_labels
        ),
        InputOrder.both: partial(
            _add_both, degrade_directional=degrade_directional_labels
        ),
    }

    stats = []
    with Progress() as progress:
        for ds_config in cfg.benchmark_data:
            builder = new_benchmark_data_factory(ds_config.dataset, root_dir)
            benchmark_data = builder.load_data().create()
            ds_dir = root_dir / ds_config.dataset.directory
            cs = load_comparison_space(
                benchmark_data,
                ds_dir,
                ds_config.comparison_space,
                seed=cfg.seed,
            )
            cs_pair_gt = {cmp: benchmark_data.true_matches.get(cmp, 0) for cmp in cs}
            cs_refs = list(map(benchmark_data.id_table.get_all, cs))
            total_comparisons = len(cs) * len(cfg.matching)
            ds_task = progress.add_task(
                f"dataset {benchmark_data.name}", total=total_comparisons
            )

            for model_config in cfg.matching:
                model_task = progress.add_task(model_config.name, total=len(cs))
                matcher = new_matcher(
                    model_config,
                    root_dir,
                    _matcher_dataset_name(ds_config, benchmark_data.name),
                )
                graphs = {
                    order: ReferenceGraph(directed=True) for order in input_orders
                }
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
                        add_to_graph = input_order_map[InputOrder(order)]
                        model_results[order].append(add_to_graph(g, matcher, x, y, r))
                    progress.update(model_task, advance=1)
                    progress.update(ds_task, advance=1)
                graph_dir = output_path / benchmark_data.name / model_config.name
                graph_dir.mkdir(parents=True, exist_ok=True)
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

                    if _uses_binary_metrics(model_config, degrade_directional_labels):
                        true_matches = {
                            cmp for cmp, label in cs_pair_gt.items() if label > 0
                        }
                        metrics = _compute_binary_classifier_metrics(
                            true_matches, g, order
                        )
                    else:
                        metrics = _compute_multiclass_metrics(cs_pair_gt, g, order)
                    stats.append(
                        {
                            "dataset": benchmark_data.name,
                            "model": model_config.name,
                            **metrics,
                        }
                    )
    stats_df = pl.DataFrame(stats)
    stats_df.write_csv(csv_path)
