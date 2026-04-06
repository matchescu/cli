from pathlib import Path

import click

from ._cli_runtime import CommonCliOptions


@click.group("matchescu")
@click.option("-v", "--verbose", type=click.BOOL, is_flag=True, default=False)
@click.option(
    "-d",
    "--root-dir",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
    default=Path.cwd(),
    help="root directory for all commands",
)
@click.pass_context
def matchescu(ctx: click.Context, verbose: bool, root_dir: Path):
    root_dir = Path(root_dir).absolute()
    ctx.obj = CommonCliOptions(verbose=verbose, root_dir=root_dir)
