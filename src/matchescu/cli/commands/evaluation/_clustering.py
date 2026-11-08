import os
from collections.abc import Callable
from functools import partial
from pathlib import Path

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
from rich.progress import Progress

from matchescu.cli.config import (
    EvaluationBenchmarkConfig,
    EvaluationConfig,
    new_benchmark_data_factory,
)
from matchescu.cli.data import load_comparison_space, load_comparison_space_clusters
from matchescu.cli.runtime import get_options, make_absolute_path

from ._eval_opts import EvalOptions
from ._input_order import InputOrder, validate_input_order_option
from ._parallel import UnitReporter, UnitRunner, workers_option

CLUSTERING_ALGOS: dict[str, Callable[..., ClusteringAlgorithm[RefId]]] = {
    "WCC": partial(WeaklyConnectedComponents, threshold=0.0),
    "MCL": partial(MarkovClustering, threshold=0.0),
    "PC": partial(ParentCenterClustering, threshold=0.0),
    "LEI": partial(LeidenPartitioning, threshold=0.0),
    "SC": partial(SpectralClustering, detect_wcc=True, max_power_iterations=100000),
}
SEEDED_CLUSTERERS = frozenset({"LEI", "SC"})
SUPPORTED_CLUSTERERS = frozenset(CLUSTERING_ALGOS) | SEEDED_CLUSTERERS
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


def _new_clusterer(
    algo_key: str, refs: list[RefId], seed: int
) -> ClusteringAlgorithm[RefId]:
    extra_args = {}
    match algo_key:
        case "LEI":
            extra_args["seed"] = seed
        case "SC":
            extra_args["k_means_random_state"] = seed
    clustering_factory = CLUSTERING_ALGOS[algo_key]
    return clustering_factory(refs, **extra_args)


def _load_dataset_context(
    ds_config: EvaluationBenchmarkConfig, seed: int, root_dir: Path
) -> tuple[str, list[RefId], frozenset[frozenset[RefId]]]:
    ds_dir = root_dir / ds_config.dataset.directory
    builder = new_benchmark_data_factory(ds_config.dataset, root_dir)
    benchmark_data = builder.load_data().create()
    cs = load_comparison_space(
        benchmark_data, ds_dir, ds_config.comparison_space, seed=seed
    )
    all_ref_ids = list(dict.fromkeys(x for cmp in cs for x in cmp))
    cs_path = ds_dir / ds_config.comparison_space.file_name
    clusters_path = ds_dir / (
        ds_config.clusters_file_name or f"{cs_path.stem}-clusters.csv"
    )
    true_clusters = load_comparison_space_clusters(
        clusters_path, benchmark_data, cs, all_ref_ids
    )
    return benchmark_data.name, all_ref_ids, true_clusters


def _eval_clustering_unit(
    algo_key: str,
    ds_config: EvaluationBenchmarkConfig,
    matcher_names: list[str],
    input_orders: list[str],
    seed: int,
    root_dir: Path,
    graph_root: Path,
    reporter: UnitReporter,
) -> list[dict]:
    """Evaluate one (clustering algorithm, dataset) unit across all matchers/orders."""
    ds_name, all_ref_ids, true_clusters = _load_dataset_context(
        ds_config, seed, root_dir
    )
    clustering_algo = _new_clusterer(algo_key, all_ref_ids, seed)
    stats = []
    for matcher_name in matcher_names:
        for order in input_orders:
            gml_path = graph_root / ds_name / matcher_name / f"{order}-graph.gml"
            # 'load' overrides whether graph is directed or not
            graph = ReferenceGraph().load(GmlGraphPersistence(gml_path))
            er_result = clustering_algo(graph)
            metrics_dict = {
                key: metric(true_clusters, er_result) for key, metric in CLUSTER_METRICS
            }
            stats.append(
                {
                    "dataset": ds_name,
                    "algorithm": algo_key,
                    "matcher": matcher_name,
                    "input_order": order,
                    **metrics_dict,
                }
            )
            reporter.advance(1)
    return stats


@click.command("clustering")
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
@workers_option()
@click.pass_context
def evaluate_clustering(
    ctx: click.Context,
    reference_graph_dir: str | os.PathLike,
    stats_csv_path: str | os.PathLike,
    input_orders: list[InputOrder],
    workers: int | None,
):
    """Evaluate a matcher's quality in controlled settings."""
    eval_opts: EvalOptions[EvaluationConfig] = get_options(ctx)
    root_dir = eval_opts.root_dir
    cfg = eval_opts.config
    graph_root = make_absolute_path(reference_graph_dir, root_dir)
    csv_path = make_absolute_path(stats_csv_path, root_dir)
    matcher_names = [m.name for m in cfg.matching]
    orders = [str(order) for order in input_orders]

    supported = [a for a in cfg.clustering.algorithms if a in SUPPORTED_CLUSTERERS]
    unit_total = len(matcher_names) * len(orders)
    units = [
        (
            f"{algo_key}/{ds_config.dataset.directory}",
            (
                algo_key,
                ds_config,
                matcher_names,
                orders,
                cfg.seed,
                root_dir,
                graph_root,
            ),
            unit_total,
        )
        for algo_key in supported
        for ds_config in cfg.benchmark_data
    ]

    stats = []
    with Progress() as progress:
        runner = UnitRunner(workers, progress)
        for _, unit_stats in runner.run(units, _eval_clustering_unit):
            stats.extend(unit_stats)

    stats_df = pl.DataFrame(stats)
    stats_df.write_csv(csv_path)
