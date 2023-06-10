import os.path
import random
import click

from ._utils import _print


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
def generate(input_file: str, count: int, output_directory: str):
    from abstractions.data_structures import Table
    from data_generators.tabular import random_sub_tables

    original_data = Table.load_csv(input_file)
    original_col_count = len(original_data.columns)
    fixed_cols = set(random.choices(list(map(lambda x: x.name, original_data.columns)), k=original_col_count // 2))
    fixed_col_count = len(fixed_cols)
    derived_data = random_sub_tables(original_data, count, fixed_col_count+1, original_col_count, *fixed_cols)

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    input_file_name = os.path.basename(input_file)
    for i, table in enumerate(derived_data):
        out_file_name = f"{i+1:05}-sub-{input_file_name}"
        out_file_path = os.path.join(output_directory, out_file_name)
        _print("writing", out_file_path)
        table.save_csv(out_file_path)
    _print("golden master was generated")

