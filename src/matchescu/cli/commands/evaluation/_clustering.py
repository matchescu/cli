import os
from functools import partial
from pathlib import Path
from typing import cast

import click
import polars as pl
from matchescu.clustering import (
    ClusteringAlgorithm,
    LeidenPartitioning,
    MarkovClustering,
    ParentCenterClustering,
    SpectralClustering,
    WeaklyConnectedComponents,
)
from matchescu.similarity import GmlGraphPersistence, ReferenceGraph
from matchescu.typing import EntityReferenceIdentifier as RefId
from pyresolvemetrics import (
    adjusted_rand_index,
    cluster_comparison_measure,
    cluster_precision,
    cluster_recall,
    pair_comparison_measure,
    pair_precision,
    pair_recall,
    twi,
)
from rich.progress import Progress, TaskID

from matchescu.cli.config import EvaluationConfig, new_benchmark_data_factory
from matchescu.cli.data import load_comparison_space, load_comparison_space_clusters
from matchescu.cli.runtime import get_options, make_absolute_path

from ._cmd_group import EvalOptions, evaluate
from ._input_order import InputOrder, validate_input_order_option

CLUSTERING_ALGOS: dict[str, ClusteringAlgorithm[RefId]] = {
    "WCC": WeaklyConnectedComponents,
    "MCL": MarkovClustering,
    "PC": ParentCenterClustering,
    "LEI": LeidenPartitioning,
    "SC": cast(
        ClusteringAlgorithm,
        cast(object, partial(SpectralClustering, detect_wcc=True)),
    ),
}
CLUSTER_METRICS = [
    ("pp", pair_precision),
    ("pr", pair_recall),
    ("pc", pair_comparison_measure),
    ("cp", cluster_precision),
    ("cr", cluster_recall),
    ("cc", cluster_comparison_measure),
    ("ari", adjusted_rand_index),
    ("twi", twi),
]


def _load_reference_graphs(
    graph_root, benchmark_data, matcher_configs, input_orders
) -> dict:
    result = {}
    for match_cfg in matcher_configs:
        for order in input_orders:
            gml_path = (
                graph_root / benchmark_data.name / match_cfg.name / f"{order}-graph.gml"
            )
            # 'load' overrides whether graph is directed or not
            graph = ReferenceGraph().load(GmlGraphPersistence(gml_path))
            result.setdefault(match_cfg.name, []).append((order, graph))
    return result


def _load_dataset_matcher_data(
    cfg: EvaluationConfig | None,
    root_dir: Path | None,
    graph_root: Path,
    input_orders: list[str],
) -> dict[
    str,
    tuple[
        set[RefId],
        frozenset[frozenset[RefId]],
        dict[str, list[tuple[str, ReferenceGraph]]],
    ],
]:
    dataset_matcher_data = {}
    for ds_config in cfg.benchmark_data:
        ds_dir = root_dir / ds_config.dataset.directory
        builder = new_benchmark_data_factory(ds_config.dataset, root_dir)
        benchmark_data = builder.load_data().create()
        cs = load_comparison_space(benchmark_data, ds_dir, ds_config.comparison_space)
        all_ref_ids = {x for cmp in cs for x in cmp}
        cs_path = ds_dir / ds_config.comparison_space.file_name

        clusters_path = ds_dir / (
            ds_config.clusters_file_name or f"{cs_path.stem}-clusters.csv"
        )
        true_clusters = load_comparison_space_clusters(
            clusters_path, benchmark_data, cs, all_ref_ids
        )
        reference_graphs = _load_reference_graphs(
            graph_root, benchmark_data, cfg.matching, input_orders
        )
        dataset_matcher_data.setdefault(
            benchmark_data.name, (all_ref_ids, true_clusters, reference_graphs)
        )
    return dataset_matcher_data


def _compute_metrics(
    true_clusters: frozenset[frozenset[RefId]],
    algorithm: ClusteringAlgorithm[RefId],
    graph: ReferenceGraph,
    progress: Progress,
    task: TaskID,
    desc: str,
) -> dict:
    er_result = algorithm(graph)
    progress.advance(task)
    metrics_dict = {}
    for key, metric in CLUSTER_METRICS:
        progress.update(task_id=task, description=f"{key} {desc}")
        metrics_dict.setdefault(key, metric(true_clusters, er_result))
        progress.advance(task)
    return metrics_dict


@evaluate.command("clustering")
@click.option(
    "-R",
    "--reference-graph-dir",
    required=True,
    type=click.Path(dir_okay=True, readable=True, resolve_path=True),
    help="directory containing saved reference graphs",
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
    input_orders: list[InputOrder],
):
    """Evaluate a matcher's quality in controlled settings."""
    eval_opts: EvalOptions[EvaluationConfig] = get_options(ctx)
    root_dir = eval_opts.root_dir
    cfg = eval_opts.config
    graph_root = make_absolute_path(reference_graph_dir, root_dir)
    csv_path = make_absolute_path(stats_csv_path, root_dir)
    n_orders = len(input_orders)
    n_metrics = len(CLUSTER_METRICS) + 1  # +1 for performing clustering itself
    n_datasets = len(cfg.benchmark_data)
    n_clusterers = len(cfg.clustering.algorithms)
    n_matchers = len(cfg.matching)
    total_steps = n_clusterers * n_datasets * n_matchers * n_metrics * n_orders
    dataset_matcher_data = _load_dataset_matcher_data(
        cfg, root_dir, graph_root, input_orders
    )

    stats = []
    with Progress() as progress:
        task = progress.add_task("compute cluster metrics", total=total_steps)

        for algo_key in cfg.clustering.algorithms:
            if algo_key not in CLUSTERING_ALGOS:
                progress.update(task, advance=n_datasets * n_matchers * n_metrics)
                continue

            algo_factory = CLUSTERING_ALGOS[algo_key]
            for ds_name, (
                all_ref_ids,
                true_clusters,
                reference_graphs,
            ) in dataset_matcher_data.items():
                clustering_algo = algo_factory(all_ref_ids, threshold=0.0)
                for matcher_name, graphs in reference_graphs.items():
                    for order, graph in graphs:
                        stats.append(
                            {
                                "dataset": ds_name,
                                "algorithm": algo_key,
                                "matcher": matcher_name,
                                "input_order": order,
                                **_compute_metrics(
                                    true_clusters,
                                    clustering_algo,
                                    graph,
                                    progress,
                                    task,
                                    f"{algo_key}({order}) - {ds_name}/{matcher_name}",
                                ),
                            }
                        )

    stats_df = pl.DataFrame(stats)
    stats_df.write_csv(csv_path)
