import os

import click
import polars as pl
from rich.progress import Progress

from matchescu.cli.config import EvaluationConfig, new_benchmark_data_factory
from matchescu.cli.data import load_comparison_space
from matchescu.cli.models import new_matcher
from matchescu.cli.runtime import get_options, make_absolute_path
from matchescu.similarity import ReferenceGraph, GmlGraphPersistence
from pyresolvemetrics import precision, recall, f1

from ._cmd_group import evaluate, EvalOptions


def _compute_metrics(
    true_matches: set, graph: ReferenceGraph, input_order: str
) -> dict:
    actual = set(
        graph.matches(min_weight=0.0)
    )  # ensures we don't filter out any matches
    return {
        "input_order": input_order,
        "precision": precision(true_matches, actual),
        "recall": recall(true_matches, actual),
        "f1": f1(true_matches, actual),
    }


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
@click.pass_context
def main(
    ctx: click.Context,
    reference_graph_dir: str | os.PathLike,
    stats_csv_path: str | os.PathLike,
):
    """Evaluate a matcher's quality in controlled settings."""
    eval_opts: EvalOptions[EvaluationConfig] = get_options(ctx)
    root_dir = eval_opts.root_dir
    cfg = eval_opts.config
    output_path = make_absolute_path(reference_graph_dir, root_dir)
    csv_path = make_absolute_path(stats_csv_path, root_dir)

    stats = []
    with Progress() as progress:
        for ds_config in cfg.benchmark_data:
            builder = new_benchmark_data_factory(ds_config.dataset, root_dir)
            benchmark_data = builder.load_data().create()
            ds_dir = root_dir / ds_config.dataset.directory
            cs = load_comparison_space(
                benchmark_data, ds_dir, ds_config.comparison_space
            )
            cs_true_matches = set(
                cmp for cmp in cs if cmp in benchmark_data.true_matches
            )
            cs_refs = list(map(benchmark_data.id_table.get_all, cs))
            total_comparisons = len(cs) * len(cfg.matching)
            ds_task = progress.add_task(
                f"dataset {benchmark_data.name}", total=total_comparisons
            )

            for model_config in cfg.matching:
                model_task = progress.add_task(model_config.name, total=len(cs))
                matcher = new_matcher(model_config, root_dir, benchmark_data.name)
                graph = ReferenceGraph(matcher, directed=True)
                for x, y in cs_refs:
                    graph.add(x, y)
                    progress.update(model_task, advance=1)
                    progress.update(ds_task, advance=1)
                graph_dir = output_path / benchmark_data.name / model_config.name
                graph_dir.mkdir(parents=True, exist_ok=True)
                graph.save(GmlGraphPersistence(graph_dir / "graph.gml"))
                stats.append(
                    {
                        "dataset": benchmark_data.name,
                        "model": model_config.name,
                        **_compute_metrics(
                            cs_true_matches, graph, input_order="forward"
                        ),
                    }
                )
    stats_df = pl.DataFrame(stats)
    stats_df.write_csv(csv_path)
