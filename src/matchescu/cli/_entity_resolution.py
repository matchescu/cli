import json
from math import isnan
from numbers import Number

import click
import pandas
from matchescu.adt.entity_resolution_result import EntityResolutionResult

from matchescu.entity_matchers import ppjoin_adapter
from matchescu.instrumentation.timer import timer
from matchescu.json import MatchescuEncoder


def _cleanup(value):
    if isinstance(value, Number):
        if isnan(value):
            return ""
    if value is None:
        return ""
    return str(value)


def _read_csv(file_path: str) -> pandas.DataFrame:
    df = pandas.read_csv(file_path, encoding_errors="ignore")
    try:
        column_names = df.columns.tolist()
        idx = column_names.index("id")
        if idx < len(column_names) - 1:
            # move id column at the end
            cols = column_names[-idx + 1 :] + column_names[: -idx + 1]
            df = df[cols]
    except ValueError:
        pass

    return df.applymap(_cleanup)


@click.command("entity-resolution")
@click.option(
    "-i",
    "--input-file",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True
    ),
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
    required=True,
)
def main(input_file: list[str], threshold: float, output_file: str):
    _, er_result = match_entities(threshold, input_file)
    with open(output_file, "w") as fd:
        json.dump(
            {
                "fsm": er_result.fsm,
                "algebraic": er_result.algebraic,
                "serf": er_result.serf,
            },
            fd,
            indent=2,
            cls=MatchescuEncoder,
        )


@timer("match-entities")
def match_entities(
    threshold: float, input_file: list[str]
) -> tuple[float, EntityResolutionResult]:
    data_frames = [df for df in map(_read_csv, input_file)]
    return threshold, ppjoin_adapter(data_frames, threshold)
