import os

import click
import polars as pl
from rich.progress import Progress

from matchescu.cli.config import EvaluationConfig, new_benchmark_data_factory
from matchescu.cli.data import load_comparison_space
from matchescu.cli.models import new_matcher
from matchescu.cli.runtime import get_options, make_absolute_path
from matchescu.matching import Matcher
from matchescu.similarity import ReferenceGraph, GmlGraphPersistence
from matchescu.typing import EntityReference
from pyresolvemetrics import precision, recall, f1
from sklearn.metrics import matthews_corrcoef

from ._cmd_group import evaluate, EvalOptions
from ._input_order import InputOrder, validate_input_order_option


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


def _add_normal(
    g: ReferenceGraph, matcher: Matcher, x: EntityReference, y: EntityReference
):
    g.add(matcher(x, y))


def _add_reverse(
    g: ReferenceGraph, matcher: Matcher, x: EntityReference, y: EntityReference
):
    g.add(matcher(y, x))


def _add_both(
    g: ReferenceGraph, matcher: Matcher, x: EntityReference, y: EntityReference
):
    g.add(matcher(x, y))
    g.add(matcher(y, x))


@evaluate.command("matching")
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
@click.pass_context
def main(
    ctx: click.Context,
    reference_graph_dir: str | os.PathLike,
    stats_csv_path: str | os.PathLike,
    input_orders: list[str],
):
    """Evaluate a matcher's quality in controlled settings."""
    eval_opts: EvalOptions[EvaluationConfig] = get_options(ctx)
    root_dir = eval_opts.root_dir
    cfg = eval_opts.config
    output_path = make_absolute_path(reference_graph_dir, root_dir)
    csv_path = make_absolute_path(stats_csv_path, root_dir)
    input_order_map = {
        InputOrder.normal: _add_normal,
        InputOrder.reverse: _add_reverse,
        InputOrder.both: _add_both,
    }

    stats = []
    with Progress() as progress:
        for ds_config in cfg.benchmark_data:
            builder = new_benchmark_data_factory(ds_config.dataset, root_dir)
            benchmark_data = builder.load_data().create()
            ds_dir = root_dir / ds_config.dataset.directory
            cs = load_comparison_space(
                benchmark_data, ds_dir, ds_config.comparison_space
            )
            cs_pair_gt = {cmp: benchmark_data.true_matches.get(cmp, 0) for cmp in cs}
            cs_refs = list(map(benchmark_data.id_table.get_all, cs))
            total_comparisons = len(cs) * len(cfg.matching)
            ds_task = progress.add_task(
                f"dataset {benchmark_data.name}", total=total_comparisons
            )

            for model_config in cfg.matching:
                model_task = progress.add_task(model_config.name, total=len(cs))
                matcher = new_matcher(model_config, root_dir, benchmark_data.name)
                graphs = {
                    order: ReferenceGraph(directed=True) for order in input_orders
                }
                for x, y in cs_refs:
                    for order, g in graphs.items():
                        add_to_graph = input_order_map[InputOrder(order)]
                        add_to_graph(g, matcher, x, y)
                    progress.update(model_task, advance=1)
                    progress.update(ds_task, advance=1)
                graph_dir = output_path / benchmark_data.name / model_config.name
                graph_dir.mkdir(parents=True, exist_ok=True)
                for order, g in graphs.items():
                    g.save(GmlGraphPersistence(graph_dir / f"{order}-graph.gml"))
                    if model_config.type != "multiclass":
                        true_matches = set(
                            cmp for cmp, label in cs_pair_gt.items() if label > 0
                        )
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
