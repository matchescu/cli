import os
from dataclasses import asdict, dataclass, field
from typing import Generic

import click

from matchescu.cli._cmd_group import matchescu
from matchescu.cli.config import EvaluationConfig, JSONConfig, TConfig
from matchescu.cli.runtime import CommonCliOptions, get_options, make_absolute_path


@dataclass(frozen=True)
class EvalOptions(CommonCliOptions, Generic[TConfig]):
    config: TConfig | None = field(default=None)


@matchescu.group("evaluate")
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
def evaluate(ctx: click.Context, config_file: str | os.PathLike):
    """Evaluation commands."""
    options = get_options(ctx)
    config_path = make_absolute_path(config_file, options.root_dir)
    cfg = JSONConfig(config_path, EvaluationConfig).load()
    ctx.obj = EvalOptions(config=cfg.config_obj, **asdict(options))
