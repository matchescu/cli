import json

import click
import pandas

from matchescu.entity_matchers import ppjoin_adapter
from matchescu.json import MatchescuEncoder


def _read_csv(file_path: str) -> pandas.DataFrame:
    with open(file_path, "r") as fd:
        return pandas.read_csv(fd)


@click.command("entity-resolution")
@click.option(
    "-i",
    "--input-file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True),
    required=True,
    multiple=True,
)
@click.option(
    "-t",
    "--threshold",
    type=click.FLOAT,
    required=True,
    default=0,
)
@click.option(
    "-o",
    "--output-file",
    type=click.Path(file_okay=True, dir_okay=False, writable=True, resolve_path=True),
    required=True
)
def match_entities(input_file: list[str], threshold: float, output_file: str):
    data_frames = [df for df in map(_read_csv, input_file)]
    er_result = ppjoin_adapter(data_frames)
    with open(output_file, "w") as fd:
        json.dump({
            "fsm": er_result.fsm,
        }, fd, indent=2, cls=MatchescuEncoder)
