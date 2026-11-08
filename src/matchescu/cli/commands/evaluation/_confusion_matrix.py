import os
from pathlib import Path

import click
import polars as pl
from rich.progress import Progress
from sklearn.metrics import confusion_matrix

from matchescu.cli.config import EvaluationConfig, new_benchmark_data_factory
from matchescu.cli.data import (
    count_comparison_space_rows,
    load_comparison_space,
    load_reference_graph,
)
from matchescu.cli.runtime import get_options, make_absolute_path

from ._eval_opts import EvalOptions
from ._input_order import InputOrder, validate_input_order_option
from ._parallel import UnitReporter, UnitRunner, workers_option
from ._stale import REBUILD_FROM_MATCHING, abort_on_stale_graph

_LABEL_NAMES = {
    0: "non-match",
    1: "full match",
    2: "forward-only",
    3: "reverse-only",
}


def _confusion_matrix_unit(
    gml_path: Path,
    y_true: list[int],
    comparisons: list[tuple],
    ds_name: str,
    matcher_name: str,
    input_order: str,
    reporter: UnitReporter,
) -> list[dict]:
    """Compute the confusion matrix rows for one (dataset, matcher, order) unit."""
    graph = load_reference_graph(gml_path, comparisons)
    y_pred = [graph.label(x, y) for x, y in comparisons]
    reporter.advance(len(comparisons))
    true_labels = sorted({0, *y_true})
    matrix = confusion_matrix(y_true, y_pred, labels=true_labels)
    rows = []
    for i, true_num in enumerate(true_labels):
        for j, pred_num in enumerate(true_labels):
            rows.append(
                {
                    "dataset": ds_name,
                    "model": matcher_name,
                    "input_order": input_order,
                    "true_label": _LABEL_NAMES[true_num],
                    "pred_label": _LABEL_NAMES[pred_num],
                    "count": int(matrix[i, j]),
                }
            )
    return rows


@click.command("confusion-matrix")
@click.option(
    "-R",
    "--reference-graph-dir",
    "graph_root_dir",
    required=True,
    type=click.Path(dir_okay=True, readable=True, resolve_path=True),
    help="directory containing saved reference graphs",
)
@click.option(
    "-o",
    "--output-csv-path",
    "output_file",
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
@workers_option()
@click.pass_context
def evaluate_confusion_matrix(
    ctx: click.Context,
    graph_root_dir: str | os.PathLike,
    output_file: str | os.PathLike,
    input_orders: list[InputOrder],
    workers: int | None,
):
    """Evaluate a matcher's quality in controlled settings."""
    eval_opts: EvalOptions[EvaluationConfig] = get_options(ctx)
    root_dir = eval_opts.root_dir
    cfg = eval_opts.config
    graph_root_dir = make_absolute_path(graph_root_dir, root_dir)
    output_path = make_absolute_path(output_file, root_dir)
    orders = [str(order) for order in input_orders]

    per_ds_count = {
        ds_config.dataset.directory: count_comparison_space_rows(
            root_dir / ds_config.dataset.directory, ds_config.comparison_space
        )
        for ds_config in cfg.benchmark_data
    }
    units = []
    for ds_config in cfg.benchmark_data:
        ds_dir = root_dir / ds_config.dataset.directory
        builder = new_benchmark_data_factory(ds_config.dataset, root_dir)
        benchmark_data = builder.load_data().create()
        cs = load_comparison_space(
            benchmark_data,
            ds_dir,
            ds_config.comparison_space,
            seed=cfg.seed,
        )
        y_true = [benchmark_data.true_matches.get(cmp, 0) for cmp in cs]
        comparisons = list(cs)
        for match_cfg in cfg.matching:
            for order in orders:
                gml_path = (
                    graph_root_dir
                    / benchmark_data.name
                    / match_cfg.name
                    / f"{order}-graph.gml"
                )
                units.append(
                    (
                        f"{benchmark_data.name}/{match_cfg.name}/{order}",
                        (
                            gml_path,
                            y_true,
                            comparisons,
                            benchmark_data.name,
                            match_cfg.name,
                            order,
                        ),
                        per_ds_count[ds_config.dataset.directory],
                    )
                )

    cm_rows = []
    with Progress() as progress:
        runner = UnitRunner(workers, progress)
        with abort_on_stale_graph(REBUILD_FROM_MATCHING):
            for _, rows in runner.run(units, _confusion_matrix_unit):
                cm_rows.extend(rows)

    confusion_matrix_df = pl.DataFrame(cm_rows)
    confusion_matrix_df.write_csv(output_path)
