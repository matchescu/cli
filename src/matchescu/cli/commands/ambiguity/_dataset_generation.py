from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import polars as pl
from matchescu.reference_store.id_table import IdTable

from ._metadata import GenerationIdentity, MetadataStore
from ._record_similarity import RecordSimilarity
from ._variant import AmbiguousVariant
from ._variant_profiler import VariantProfiler
from ._writer import VariantNaming, VariantWriter
from .profiling import ProfileOptions

if TYPE_CHECKING:
    from ._generate import AmbiguityGenerator


class DatasetGeneration:
    """Generate, write and describe the ambiguous variants of one dataset.

    Every run writes the variants, upserts one metadata row per variant into
    the shared ``metadata.csv`` of the output directory and writes the bridge
    components table. Bridges, and therefore components, are the same for
    every alpha, so the components table is written once per run.

    :param output_dir: root directory of all generated files.
    :param options: settings of the dataset profiles.
    """

    METADATA_FILE: ClassVar[str] = "metadata.csv"

    def __init__(self, output_dir: Path, options: ProfileOptions) -> None:
        self.__output_dir = output_dir
        self.__options = options
        self.__store = MetadataStore(output_dir / self.METADATA_FILE)

    def run(self, generator: "AmbiguityGenerator", id_table: IdTable) -> None:
        """Generate every variant of `generator` and record its metadata.

        :param generator: generator configured for one source dataset.
        :param id_table: original records of that dataset.
        :raises ValueError: when the generator has no ambiguity targets; the
            direction metrics compare records on those fields only. Raised
            before anything is generated or written.
        """
        settings = generator.settings
        profiler = VariantProfiler(
            id_table, RecordSimilarity(settings.ambiguity_targets), self.__options
        )
        naming = VariantNaming(settings.dataset_name, settings.target_links)
        variants = generator()
        paths = VariantWriter(id_table, self.__output_dir, naming).write(variants)
        self.__store.upsert(
            [
                GenerationIdentity(settings, alpha, naming, paths[alpha]).as_row(
                    self.__output_dir
                )
                | profiler.profile(variant)
                for alpha, variant in variants.items()
            ]
        )
        self.__write_components(profiler, naming, variants)

    def __write_components(
        self,
        profiler: VariantProfiler,
        naming: VariantNaming,
        variants: Mapping[float, AmbiguousVariant],
    ) -> None:
        rows = profiler.components(next(iter(variants.values()))).rows()
        if rows:
            pl.DataFrame(rows).write_csv(
                self.__output_dir / naming.bridge_components, include_header=True
            )
