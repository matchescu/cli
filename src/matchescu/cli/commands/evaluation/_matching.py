import click
from matchescu.cli._cli_runtime import get_options
from ._cmd_group import evaluate


@evaluate.command("matching")
@click.option(
    "-f",
    "--config-file",
    required=True,
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True
    ),
    help="configuration file for the evaluation pipeline",
)
@click.pass_context
def main(ctx: click.Context):
    """Evaluate a matcher's quality in controlled settings."""
    root_dir = get_options(ctx).root_dir
    print(root_dir)
