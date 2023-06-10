import json
import os
from typing import Generator

import click

from ._utils import _print, MatchescuJSONEncoder


def _get_file_paths(input_dir: str) -> Generator[str, None, None]:
    for dir_path, __, file_names in os.walk(input_dir, True):
        for file_name in file_names:
            if not file_name.endswith(".csv"):
                continue
            fpath = os.path.join(dir_path, file_name)
            _print(fpath)
            yield fpath


@click.command("entity-resolution")
@click.option(
    "-i",
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True),
    required=True,
)
@click.option(
    "-t",
    "--threshold",
    type=click.FLOAT,
    required=True,
    default=0.6,
)
@click.option(
    "-o",
    "--output-file",
    type=click.Path(file_okay=True, dir_okay=False, writable=True, resolve_path=True),
    required=True
)
def match_entities(input_dir: str, threshold: float, output_file: str):
    from abstractions.data_structures import Table
    from entity_matchers.ppjoin import find_duplicates_across

    tables = [
        Table.load_csv(file_path=path) for path in _get_file_paths(input_dir)
    ]
    datasets = [
        [row.values for row in table]
        for table in tables
    ]
    duplicates = find_duplicates_across(datasets, threshold)
    with open(output_file, "w") as result:
        result.write(json.dumps(duplicates, cls=MatchescuJSONEncoder, indent=4))
