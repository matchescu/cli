import os
from os import PathLike
from logging import getLogger
from pathlib import Path

import polars as pl

from matchescu.matching.evaluation.data.benchmark import BenchmarkData
from matchescu.matching.evaluation.ground_truth import read_clusters_csv
from matchescu.reference_store.comparison_space import BinaryComparisonSpace
from matchescu.typing import EntityReferenceIdentifier as RefId

_log = getLogger(__file__)


def load_comparison_space_clusters(
    path: str | PathLike,
    data: BenchmarkData,
    cs: BinaryComparisonSpace,
    all_ref_ids: set[RefId],
) -> frozenset[frozenset[RefId]]:
    """Load the clusters saved by ``GroundTruthComparisonSpaceGenerator``.

    :param path: CSV file containing the clustering information
    :param data: if path doesn't exist, generate clusters from the ground truth
        provided by this parameter
    :param cs: if path doesn't exist, generate clusters for the comparisons
        specified by this parameter
    :param all_ref_ids: a set of all reference identifiers, including those that
        aren't part of clusters and should be treated as singletons
    """
    path = Path(path)
    max_id = 0
    if path.exists():
        clusters = read_clusters_csv(
            path,
            has_header=True,
            id_col=0,
            source_col=1,
            label_col=2,
        )
        clustered_refs = set(
            ref_id for cluster in clusters.values() for ref_id in cluster
        )
        max_id = max(clusters.keys())
    else:
        df = downscale_clusters(data, cs, path)
        clusters = {}
        clustered_refs = set()
        for row in df.iter_rows(named=True):
            ref_id = RefId(label=row["id"], source=row["source"])
            cluster_id = int(row["cluster_id"])
            max_id = max(cluster_id, max_id)
            clusters.setdefault(cluster_id, set()).add(ref_id)
            clustered_refs.add(ref_id)
    singletons = all_ref_ids - clustered_refs
    for ref_id in singletons:
        clusters.setdefault(max_id + 1, set()).add(ref_id)
        max_id += 1
    return frozenset(frozenset(cluster) for cluster_no, cluster in clusters.items())


def downscale_clusters(
    data: BenchmarkData,
    comparison_space: BinaryComparisonSpace,
    clusters_output_path: str | PathLike | None = None,
) -> pl.DataFrame:
    clusters: dict[int, set[RefId]] = {}
    for tpl in comparison_space:
        if tpl not in data.true_matches:
            continue
        for ref_id in tpl:
            if (cluster_no := data.ref_id_cluster_map.get(ref_id)) is None:
                continue
            clusters.setdefault(cluster_no, set()).add(ref_id)
    cluster_data = []

    # renumber the clusters and skip singletons
    for cluster_no, cluster in enumerate(
        (c for c in clusters.values() if len(c) > 1), start=1
    ):
        for ref_id in cluster:
            cluster_data.append(
                {
                    "id": ref_id.label,
                    "source": ref_id.source,
                    "cluster_id": cluster_no,
                }
            )
    clusters_df = pl.DataFrame(cluster_data).sort(by="cluster_id")
    _write_to_file(clusters_df, clusters_output_path)
    return clusters_df


def _write_to_file(df: pl.DataFrame, path: str | PathLike | None) -> None:
    if path is None:
        return
    if os.path.isdir(path):
        _log.warning("'%s' is a directory", path)
        return
    if os.access(path, os.F_OK) and not os.access(path, os.W_OK):
        _log.warning("can't write to file '%s'", path)
        return

    _log.info("saving clusters: %d clusters", df.n_unique(subset=["cluster_id"]))
    df.write_csv(path, include_header=True)
