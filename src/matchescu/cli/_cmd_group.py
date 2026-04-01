import click

from ._cli_runtime import CommonCliOptions


@click.group("matchescu")
@click.option("-v", "--verbose", type=click.BOOL, is_flag=True, default=False)
@click.pass_context
def matchescu(ctx: click.Context, verbose: bool):
    ctx.obj = CommonCliOptions(verbose=verbose)