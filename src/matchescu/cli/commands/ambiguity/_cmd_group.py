import click

from ._generate import generate_ambiguous_datasets


@click.group(
    "ambiguity", help="Commands for analysing and generating ambiguous datasets."
)
def ambiguity():
    pass


ambiguity.add_command(generate_ambiguous_datasets, "generate")
