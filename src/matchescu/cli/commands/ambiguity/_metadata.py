import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from ._settings import GenerationSettings
from ._writer import VariantNaming, VariantPaths

type MetadataRow = Mapping[str, Any]


@dataclass(frozen=True)
class GenerationIdentity:
    """Identity columns of one generated variant in the metadata table.

    One row describes one alpha variant in both its published forms, so the
    row is keyed by the undirected variant name and lists both directories.

    :ivar settings: effective generator parameters.
    :ivar alpha: blend weight of the variant.
    :ivar naming: names of the variant's files and directories.
    :ivar paths: directories the variant was written to.
    """

    LIST_SEPARATOR: ClassVar[str] = ","

    settings: GenerationSettings
    alpha: float
    naming: VariantNaming
    paths: VariantPaths

    def as_row(self, root: Path) -> dict[str, Any]:
        """Return the identity columns, with paths relative to `root`.

        :param root: directory the metadata table lives in.
        :returns: column name to value, ``dataset`` first.
        """
        return {
            "dataset": self.naming.undirected(self.alpha),
            "source_dataset": self.settings.dataset_name,
            "alpha": self.alpha,
            "tag": self.naming.tag(self.alpha),
            "target_floor": self.settings.target_floor,
            "target_links": self.settings.target_links,
            "min_ambiguity": self.settings.min_ambiguity,
            "ambiguity_targets": self.LIST_SEPARATOR.join(
                self.settings.ambiguity_targets
            ),
            "secondary_features": self.LIST_SEPARATOR.join(
                self.settings.secondary_features
            ),
            "max_length_ratio": self.settings.max_length_ratio,
            **self.settings.pairing.as_row(),
            "undirected_path": str(self.paths.undirected.relative_to(root)),
            "directed_path": str(self.paths.directed.relative_to(root)),
        }


class MetadataStore:
    """CSV table of dataset metadata, updated row by row.

    Rows are identified by their ``dataset`` column. Upserting replaces rows
    in place and appends new ones, so generating datasets one by one into
    the same directory accumulates a single table. Columns keep their first
    seen order; columns new to the table are appended.

    :param path: location of the CSV file; created on the first upsert.
    """

    KEY: ClassVar[str] = "dataset"

    def __init__(self, path: Path) -> None:
        self.__path = path

    def upsert(self, rows: Sequence[MetadataRow]) -> None:
        """Insert or replace rows by their ``dataset`` value.

        :param rows: rows to store; ``None`` values become empty cells.
        :raises ValueError: when a row has no ``dataset`` value.
        """
        stored = {row[self.KEY]: dict(row) for row in self.__read()}
        for row in rows:
            stored[self.__key(row)] = dict(row)
        self.__write(list(stored.values()))

    def __key(self, row: MetadataRow) -> Any:
        if row.get(self.KEY) is None:
            raise ValueError(f"metadata row has no '{self.KEY}' value: {row}")
        return row[self.KEY]

    def __read(self) -> list[dict[str, str]]:
        if not self.__path.exists():
            return []
        with self.__path.open(newline="") as handle:
            return list(csv.DictReader(handle))

    def __write(self, rows: list[dict[str, Any]]) -> None:
        columns = list(dict.fromkeys(column for row in rows for column in row))
        with self.__path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, restval="")
            writer.writeheader()
            writer.writerows(rows)
