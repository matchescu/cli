import os
from functools import partial
from pathlib import Path
from typing import cast

import click
import polars as pl
from rich.progress import Progress, TaskID

from matchescu.cli.config import EvaluationConfig, new_benchmark_data_factory
from matchescu.cli.data import load_comparison_space_clusters, load_comparison_space
from matchescu.cli.models import new_matcher
from matchescu.cli.runtime import get_options, make_absolute_path
from matchescu.clustering import (
    ClusteringAlgorithm,
    WeaklyConnectedComponents,
    MarkovClustering,
    ParentCenterClustering,
    LeidenPartitioning,
    SpectralClustering,
)
from matchescu.similarity import ReferenceGraph, GmlGraphPersistence
from matchescu.typing import EntityReference, EntityReferenceIdentifier as RefId
from pyresolvemetrics import (
    pair_precision,
    pair_recall,
    pair_comparison_measure,
    cluster_precision,
    cluster_recall,
    cluster_comparison_measure,
    adjusted_rand_index,
    twi,
)


from ._cmd_group import evaluate, EvalOptions

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
    root_dir, graph_root, benchmark_data, matcher_configs
) -> dict:
    result = {}
    for match_cfg in matcher_configs:
        gml_path = graph_root / benchmark_data.name / match_cfg.name / "graph.gml"
        matcher = new_matcher(match_cfg, root_dir, benchmark_data.name)
        # 'load' overrides whether graph is directed or not
        graph = ReferenceGraph(matcher).load(GmlGraphPersistence(gml_path))
        result.setdefault(match_cfg.name, graph)
    return result


def _load_dataset_matcher_data(
    cfg: EvaluationConfig | None, root_dir: Path | None, graph_root: Path
) -> dict[
    str,
    tuple[
        set[RefId],
        frozenset[frozenset[RefId]],
        dict[str, ReferenceGraph[EntityReference]],
    ],
]:
    dataset_matcher_data = {}
    for ds_config in cfg.benchmark_data:
        ds_dir = root_dir / ds_config.dataset.directory
        builder = new_benchmark_data_factory(ds_config.dataset, root_dir)
        benchmark_data = builder.load_data().create()
        cs = load_comparison_space(benchmark_data, ds_dir, ds_config.comparison_space)
        all_ref_ids = set(x for cmp in cs for x in cmp)
        cs_path = ds_dir / ds_config.comparison_space.file_name

        clusters_path = ds_dir / (
            ds_config.clusters_file_name or f"{cs_path.stem}-clusters.csv"
        )
        true_clusters = load_comparison_space_clusters(
            clusters_path, benchmark_data, cs, all_ref_ids
        )
        reference_graphs = _load_reference_graphs(
            root_dir, graph_root, benchmark_data, cfg.matching
        )
        dataset_matcher_data.setdefault(
            benchmark_data.name, (all_ref_ids, true_clusters, reference_graphs)
        )
    return dataset_matcher_data


def _compute_metrics(
    true_clusters: frozenset[frozenset[RefId]],
    algorithm: ClusteringAlgorithm[RefId],
    graph: ReferenceGraph[EntityReference],
    progress: Progress,
    task: TaskID,
) -> dict:
    er_result = algorithm(graph)
    progress.advance(task)
    metrics_dict = {}
    for key, metric in CLUSTER_METRICS:
        progress.update(task_id=task, description=f"computing {key}")
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
    graph_root = make_absolute_path(reference_graph_dir, root_dir)
    csv_path = make_absolute_path(stats_csv_path, root_dir)
    n_metrics = len(CLUSTER_METRICS) + 1  # +1 for performing clustering itself
    n_datasets = len(cfg.benchmark_data)
    n_clusterers = len(cfg.clustering.algorithms)
    n_matchers = len(cfg.matching)
    total_steps = n_clusterers * n_datasets * n_matchers * n_metrics
    dataset_matcher_data = _load_dataset_matcher_data(cfg, root_dir, graph_root)

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
                for matcher_name, graph in reference_graphs.items():
                    clustering_algo = algo_factory(all_ref_ids, threshold=0.0)
                    stats.append(
                        {
                            "dataset": ds_name,
                            "algorithm": algo_key,
                            "matcher": matcher_name,
                            **_compute_metrics(
                                true_clusters,
                                clustering_algo,
                                graph,
                                progress,
                                task,
                            ),
                        }
                    )

    stats_df = pl.DataFrame(stats)
    stats_df.write_csv(csv_path)
