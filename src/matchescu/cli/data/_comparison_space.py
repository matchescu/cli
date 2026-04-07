from pathlib import Path
from typing import Iterable

from matchescu.cli.config._config import ComparisonSpaceConfig
from matchescu.comparison_space.persistence import CsvPersistence
from matchescu.matching.evaluation.ground_truth import read_clusters_csv
from matchescu.matching.evaluation.data.benchmark import BenchmarkData
from matchescu.matching.evaluation.data.generation import (
    GroundTruthComparisonSpaceGenerator,
)
from matchescu.reference_store.comparison_space import BinaryComparisonSpace
from matchescu.typing import EntityReferenceIdentifier as RefId


def _get_excluded(
    data: BenchmarkData, data_dir: Path, cs_config: ComparisonSpaceConfig
) -> Iterable[tuple[RefId, RefId]]:
    if not cs_config.excluded_files:
        yield from ()
    for file in cs_config.excluded_files:
        path = data_dir / file
        if not path.exists() or not path.is_file():
            continue
        yield from CsvPersistence(path).read(cs_config.to_csv_params(data.name))


def load_comparison_space(
    data: BenchmarkData, data_dir: Path, cs_config: ComparisonSpaceConfig
) -> BinaryComparisonSpace:
    """Load a comparison space from a CSV file or generate one.

    The comparison space of the ``data`` parameter may be too large for pragmatic
    usage. This function allows generating a comparison space using the ground
    truth provided by the ``data`` based on the parameters provided in ``cs_config``.

    Once the generation is complete, the comparison space will be saved to the
    file specified in the ``cs_config``, under the ``data_dir`` folder. This
    method also generates a CSV file containing the true clusters provided in
    the ``data``, scaled down to the comparisons in the generated comparison
    space.

    :param data: the data to generate the comparison space from.
    :param data_dir: the root directory for persistence operations related to
        the comparison space
    :param cs_config: configuration parameters for comparison space generation
        and persistence.
    """
    file_path = data_dir / cs_config.file_name
    if file_path.exists():
        persistence = CsvPersistence(file_path)
        return persistence.read(cs_config.to_csv_params(data.name))
    else:
        csg = GroundTruthComparisonSpaceGenerator(
            data.id_table,
            data.true_matches,
            data.compute_clusters(),
            neg_pos_ratio=cs_config.neg_pos_ratio,
            match_bridge_ratio=cs_config.match_bridge_ratio,
            max_total_samples=cs_config.sample_count,
            save_comparisons=True,
            save_clusters=True,
            excluded=_get_excluded(data, data_dir, cs_config),
        )
        cluster_file_path = file_path.parent / f"{file_path.stem}-clusters.csv"
        return csg(file_path, cluster_file_path)


def load_comparison_space_clusters(csv_path: Path) -> frozenset[frozenset[RefId]]:
    """Load the clusters saved by ``GroundTruthComparisonSpaceGenerator``.

    :param csv_path: CSV file containing the clustering information
    """
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    clusters = read_clusters_csv(
        csv_path,
        has_header=True,
        id_col=0,
        source_col=1,
        label_col=2,
    )
    return frozenset(frozenset(cluster) for cluster_no, cluster in clusters.items())
