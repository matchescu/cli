from collections.abc import Iterable
from functools import cache, partial
from pathlib import Path

from matchescu.comparison_space.persistence import CsvPersistence
from matchescu.matching.evaluation.data.benchmark import BenchmarkData
from matchescu.matching.evaluation.data.generation import (
    GroundTruthComparisonSpaceGenerator,
)
from matchescu.reference_store.comparison_space import BinaryComparisonSpace
from matchescu.typing import EntityReferenceIdentifier as RefId

from matchescu.cli.config._config import ComparisonSpaceConfig


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


@cache
def _label_two_bridges(data: BenchmarkData) -> frozenset[RefId]:
    return frozenset(
        right for (_, right), label in data.true_matches.items() if label > 1
    )


def _is_bridge(data: BenchmarkData, bridge_source: str | None, ref_id: RefId) -> bool:
    if bridge_source is not None:
        return ref_id.source == bridge_source
    return ref_id in _label_two_bridges(data)


def _check_bridge_source(data: BenchmarkData, bridge_source: str | None) -> None:
    if bridge_source is None:
        return
    if any(ref.id.source == bridge_source for ref in data.id_table):
        return
    raise ValueError(
        f"bridge source {bridge_source!r} does not appear in dataset {data.name!r}"
    )


def load_comparison_space(
    data: BenchmarkData,
    data_dir: Path,
    cs_config: ComparisonSpaceConfig,
    seed: int = 42,
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
    :param seed: seed for comparison space generation. Defaults to ``42``.
    """
    file_path = data_dir / cs_config.file_name
    if file_path.exists():
        persistence = CsvPersistence(file_path)
        return persistence.read(cs_config.to_csv_params(data.name))
    else:
        _check_bridge_source(data, cs_config.bridge_source)
        csg = GroundTruthComparisonSpaceGenerator(
            data.id_table,
            data.true_matches,
            data.compute_clusters(),
            neg_pos_ratio=cs_config.neg_pos_ratio,
            match_bridge_ratio=cs_config.match_bridge_ratio,
            max_total_samples=cs_config.sample_count,
            seed=seed,
            save=True,
            excluded=_get_excluded(data, data_dir, cs_config),
            is_bridge=partial(_is_bridge, data, cs_config.bridge_source),
        )
        cluster_file_path = file_path.parent / f"{file_path.stem}-clusters.csv"
        return csg(file_path, cluster_file_path)
