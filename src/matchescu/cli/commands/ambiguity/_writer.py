from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import polars as pl
from matchescu.reference_store.id_table import IdTable
from matchescu.typing import EntityReference
from matchescu.typing import EntityReferenceIdentifier as RefId

from ._variant import (
    RESERVED_BRIDGE_PROPERTIES,
    AmbiguousVariant,
    ComparisonData,
    ref_id_sort_key,
)


@dataclass(frozen=True)
class VariantPaths:
    """Directories holding the two published forms of one alpha variant.

    :ivar undirected: directory of the variant whose pairs are all labelled 1.
    :ivar directed: directory of the variant that labels target links 2.
    """

    undirected: Path
    directed: Path


class VariantNaming:
    """Name the files and directories of an ambiguous dataset's variants.

    Names encode the source dataset, the alpha weight in hundredths and, for
    more than one target link, the link count, so variants of different
    generator settings can share one output directory.

    :param dataset_name: name of the source dataset.
    :param target_links: number of target links per bridge.
    """

    PREFIX = "amb"
    DIRECTED_SUFFIX = "_dir"
    ALPHA_SCALE = 100

    def __init__(self, dataset_name: str, target_links: int) -> None:
        self.__base = f"{self.PREFIX}_{dataset_name}"
        self.__links = f"_k{target_links}" if target_links > 1 else ""

    def tag(self, alpha: float) -> str:
        """Return the alpha tag, e.g. ``a065`` for 0.65."""
        return f"a{round(alpha * self.ALPHA_SCALE):03d}"

    def undirected(self, alpha: float) -> str:
        """Return the name of the undirected variant of `alpha`."""
        return f"{self.__base}_{self.tag(alpha)}{self.__links}"

    def directed(self, alpha: float) -> str:
        """Return the name of the directed variant of `alpha`."""
        return f"{self.undirected(alpha)}{self.DIRECTED_SUFFIX}"

    @property
    def bridge_log(self) -> str:
        """File name of the bridge log; bridge text ignores target links."""
        return f"{self.__base}_bridge_log.csv"

    @property
    def link_log(self) -> str:
        """File name of the link log."""
        return f"{self.__base}{self.__links}_link_log.csv"


class VariantWriter:
    """Persist ambiguous dataset variants and their generation logs as CSV.

    Each variant is written twice, undirected and directed, as an
    ids/mapping/cluster-mapping CSV triple. The reserved bridge properties
    are stripped from the ids CSV: they document the synthesis rather than
    the published records.

    :param id_table: original records the variants extend.
    :param output_dir: root directory of every written file.
    :param naming: names of the written files and directories.
    """

    def __init__(self, id_table: IdTable, output_dir: Path, naming: VariantNaming):
        self.__id_table = id_table
        self.__output_dir = output_dir
        self.__naming = naming

    def write(
        self, variants: Mapping[float, AmbiguousVariant]
    ) -> dict[float, VariantPaths]:
        """Write every variant plus the bridge and link logs of all of them.

        :param variants: variants keyed by alpha, as the generator returns.
        :returns: the directories written for each alpha.
        """
        paths = {alpha: self.__write_variant(v) for alpha, v in variants.items()}
        self.__write_log(self.__naming.bridge_log, variants, "bridge_log")
        self.__write_log(self.__naming.link_log, variants, "link_log")
        return paths

    def __write_variant(self, variant: AmbiguousVariant) -> VariantPaths:
        ids = self.__ids_frame(variant.bridges)
        clusters = self.__cluster_frame(variant.cluster_gt)
        return VariantPaths(
            self.__write_dir(
                self.__naming.undirected(variant.alpha),
                ids,
                clusters,
                variant.undirected_gt,
            ),
            self.__write_dir(
                self.__naming.directed(variant.alpha),
                ids,
                clusters,
                variant.directed_gt,
            ),
        )

    def __write_dir(self, name, ids, clusters, mapping_gt) -> Path:
        path = self.__output_dir / name
        path.mkdir(parents=True, exist_ok=True)
        ids.write_csv(path / f"{name}_ids.csv", include_header=True)
        self.__mapping_frame(mapping_gt).write_csv(
            path / f"{name}_mapping.csv", include_header=True
        )
        clusters.write_csv(path / f"{name}_cluster_mapping.csv", include_header=True)
        return path

    def __write_log(self, file_name, variants, log_name) -> None:
        rows = list(
            chain.from_iterable(getattr(v, log_name) for v in variants.values())
        )
        if rows:
            pl.DataFrame(rows).write_csv(
                self.__output_dir / file_name, include_header=True
            )

    def __ids_frame(self, bridges: Iterable[EntityReference]) -> pl.DataFrame:
        return pl.DataFrame(
            [self.__ids_row(ref) for ref in chain(self.__id_table, bridges)]
        )

    @staticmethod
    def __ids_row(ref: EntityReference) -> dict:
        published = {
            k: v
            for k, v in ref.as_dict().items()
            if k not in RESERVED_BRIDGE_PROPERTIES
        }
        return {"id": ref.id.label, "source": str(ref.id.source), **published}

    @staticmethod
    def __cluster_frame(cluster_gt: Mapping[int, set[RefId]]) -> pl.DataFrame:
        return pl.DataFrame(
            [
                {"id": ref_id.label, "source": ref_id.source, "cluster_id": cid}
                for cid, ref_ids in sorted(cluster_gt.items())
                for ref_id in sorted(ref_ids, key=ref_id_sort_key)
            ]
        )

    @staticmethod
    def __mapping_frame(mapping_gt: Mapping[ComparisonData, int]) -> pl.DataFrame:
        return pl.DataFrame(
            [
                {
                    "left_id": left.label,
                    "left_source": left.source,
                    "right_id": right.label,
                    "right_source": right.source,
                    "label": label,
                }
                for (left, right), label in mapping_gt.items()
            ]
        )
