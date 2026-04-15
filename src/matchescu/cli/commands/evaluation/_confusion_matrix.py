import os
from pathlib import Path

import click
import polars as pl
from rich.progress import Progress
from sklearn.metrics import confusion_matrix

from matchescu.cli.config import EvaluationConfig, new_benchmark_data_factory
from matchescu.cli.data import load_comparison_space
from matchescu.cli.runtime import get_options, make_absolute_path
from matchescu.similarity import ReferenceGraph, GmlGraphPersistence

from ._cmd_group import evaluate, EvalOptions
from ._input_order import InputOrder, validate_input_order_option

_LABEL_NAMES = {
    0: "non-match",
    1: "full match",
    2: "forward-only",
    3: "reverse-only",
}


def _prepare_labels(
    cfg: EvaluationConfig | None,
    root_dir: Path | None,
    graph_root: Path,
    input_orders: list[str],
) -> dict[str, tuple[list[int], dict[str, list[int]]]]:
    dataset_matcher_data = {}
    for ds_config in cfg.benchmark_data:
        ds_dir = root_dir / ds_config.dataset.directory
        builder = new_benchmark_data_factory(ds_config.dataset, root_dir)
        benchmark_data = builder.load_data().create()
        cs = load_comparison_space(benchmark_data, ds_dir, ds_config.comparison_space)
        y_true = [benchmark_data.true_matches.get(cmp, 0) for cmp in cs]

        predictions = {}
        for match_cfg in cfg.matching:
            for order in input_orders:
                gml_path = (
                    graph_root
                    / benchmark_data.name
                    / match_cfg.name
                    / f"{order}-graph.gml"
                )
                graph = ReferenceGraph.load(GmlGraphPersistence(gml_path))
                y_pred = [graph.label(x, y) for x, y in cs]
                predictions.setdefault(match_cfg.name, []).append((order, y_pred))

        dataset_matcher_data[benchmark_data.name] = (y_true, predictions)

    return dataset_matcher_data


@evaluate.command("confusion-matrix")
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
@click.pass_context
def main(
    ctx: click.Context,
    graph_root_dir: str | os.PathLike,
    output_file: str | os.PathLike,
    input_orders: list[InputOrder],
):
    """Evaluate a matcher's quality in controlled settings."""
    eval_opts: EvalOptions[EvaluationConfig] = get_options(ctx)
    root_dir = eval_opts.root_dir
    cfg = eval_opts.config
    graph_root_dir = make_absolute_path(graph_root_dir, root_dir)
    output_path = make_absolute_path(output_file, root_dir)
    n_orders = len(input_orders)
    n_datasets = len(cfg.benchmark_data)
    n_matchers = len(cfg.matching)
    total_steps = n_datasets * n_matchers * n_orders

    labels = _prepare_labels(cfg, root_dir, graph_root_dir, input_orders)
    with Progress() as progress:
        task = progress.add_task("compute confusion matrices", total=total_steps)
        cm_rows = []
        for ds_name, (y_true, matcher_dict) in labels.items():
            true_labels = list(sorted({0, *y_true}))
            for matcher_name, input_order_predictions in matcher_dict.items():
                for input_order, y_pred in input_order_predictions:
                    # do something to store this
                    matrix = confusion_matrix(y_true, y_pred, labels=true_labels)
                    # Convert to long format
                    for i, true_num in enumerate(true_labels):
                        for j, pred_num in enumerate(true_labels):
                            cm_rows.append(
                                {
                                    "dataset": ds_name,
                                    "model": matcher_name,
                                    "input_order": input_order,
                                    "true_label": _LABEL_NAMES[true_num],
                                    "pred_label": _LABEL_NAMES[pred_num],
                                    "count": int(matrix[i, j]),
                                }
                            )
                    progress.update(task, description=f"{ds_name} / {matcher_name}")

    confusion_matrix_df = pl.DataFrame(cm_rows)
    confusion_matrix_df.write_csv(output_path)
