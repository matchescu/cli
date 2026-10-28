import click

from ._analyse import analyse_ambiguity
from ._generate import generate_ambiguous_datasets


@click.group(
    "ambiguity", help="Commands for analysing and generating ambiguous datasets."
)
def ambiguity():
    pass


ambiguity.add_command(generate_ambiguous_datasets, "generate")
ambiguity.add_command(analyse_ambiguity, "analyse")
