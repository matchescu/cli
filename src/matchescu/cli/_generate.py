import json
import os.path

import click
import pandas

from matchescu.data_generators.tables import SplitTableRandomly
from ._utils import _print
from ..json import MatchescuEncoder


@click.command("generate")
@click.option(
    "-i",
    "--input-file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True),
    required=True,
)
@click.option(
    "-o",
    "--output-directory",
    type=click.Path(file_okay=False, dir_okay=True, readable=True, resolve_path=True),
    required=True,
)
@click.option(
    "-n",
    "--count",
    type=click.INT,
    required=True,
    default=2
)
@click.option(
    "-g",
    "--gold-standard",
    type=click.Path(file_okay=True, dir_okay=False, writable=True, resolve_path=True),
    required=True,
)
@click.option(
    "-f", "--fixed", type=str, multiple=True
)
def generate(input_file: str, count: int, output_directory: str, gold_standard: str, fixed: list[str]):
    df = pandas.read_csv(input_file, header=0, encoding_errors="ignore")
    splitter = SplitTableRandomly(count, fixed)
    derived_data = splitter(df)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    input_file_name = os.path.basename(input_file)
    for i, table in enumerate(derived_data):
        out_file_name = f"{i+1:05}-sub-{input_file_name}"
        out_file_path = os.path.join(output_directory, out_file_name)
        table.to_csv(out_file_path, index=False)
    with open(gold_standard, "w") as gs_file:
        json.dump({
            "fsm": splitter.ground_truth.fsm,
            "algebraic": splitter.ground_truth.algebraic,
        }, gs_file, cls=MatchescuEncoder, indent=2)
    _print("golden master was generated")

