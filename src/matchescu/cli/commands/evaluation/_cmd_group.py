import os
from dataclasses import asdict

import click

from matchescu.cli.config import EvaluationConfig, JSONConfig
from matchescu.cli.runtime import get_options, make_absolute_path

from ._asymmetry import AsymmetryCommand
from ._clustering import evaluate_clustering
from ._confusion_matrix import evaluate_confusion_matrix
from ._eval_opts import EvalOptions
from ._matching import evaluate_matching


@click.group("evaluate")
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


evaluate.add_command(AsymmetryCommand())
evaluate.add_command(evaluate_clustering)
evaluate.add_command(evaluate_confusion_matrix)
evaluate.add_command(evaluate_matching)
