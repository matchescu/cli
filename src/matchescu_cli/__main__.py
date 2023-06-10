import click

import matchescu_cli._global_flags as flags
from ._generate import generate
from ._entity_resolution import match_entities


@click.group("matchescu")
@click.option(
    "-v",
    "--verbose",
    type=click.BOOL,
    is_flag=True,
    default=False
)
def main(verbose):
    flags.VERBOSE = verbose


if __name__ == "__main__":
    main.add_command(generate, "generate")
    main.add_command(match_entities, "entity-resolution")
    main()
