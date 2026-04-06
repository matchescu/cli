import logging
import os
from os import PathLike
from pathlib import Path

import click
import polars
import plotly.graph_objects as go
from rich.progress import Progress


from matchescu.cli.config import (
    EvaluationConfig,
    new_benchmark_data_factory,
    JSONConfig,
)
from matchescu.cli.data import load_comparison_space
from matchescu.cli.models import new_matcher
from matchescu.cli._cli_runtime import get_options
from ._cmd_group import evaluate

os.environ["DISABLE_TQDM"] = "true"


logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_parallel_utils").setLevel(logging.ERROR)


@evaluate.command("asymmetry")
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
def main(ctx: click.Context, config_file: str | PathLike):
    root_dir = get_options(ctx).root_dir
    config_file = Path(config_file).absolute()
    cfg = JSONConfig(config_file, EvaluationConfig).load()

    asymmetry = []
    with Progress() as progress:
        for ds_config in cfg.config_obj.benchmark_data:
            builder = new_benchmark_data_factory(ds_config.dataset, root_dir)
            benchmark_data = builder.load_data().create()
            ds_dir = root_dir / ds_config.dataset.directory
            cs = load_comparison_space(
                benchmark_data, ds_dir, ds_config.comparison_space
            )
            cs_refs = list(map(benchmark_data.id_table.get_all, cs))
            total_comparisons = len(cs) * len(cfg.config_obj.models)
            ds_task = progress.add_task(
                f"dataset {benchmark_data.name}", total=total_comparisons
            )

            for model_config in cfg.config_obj.models:
                model_task = progress.add_task(model_config.name, total=len(cs))
                matcher = new_matcher(model_config, root_dir, benchmark_data.name)
                model_results = []
                for x, y in cs_refs:
                    model_results.append((x.id, y.id, matcher(x, y), matcher(y, x)))
                    progress.update(ds_task, advance=1)
                    progress.update(model_task, advance=1)

                asymmetry.extend(
                    [
                        {
                            "dataset": benchmark_data.name,
                            "model": model_config.name,
                            "diff": fwd_result.label_weights[fwd_result.label]
                            - rev_result.label_weights[rev_result.label],
                            "abs_diff": abs(
                                fwd_result.label_weights[fwd_result.label]
                                - rev_result.label_weights[rev_result.label]
                            ),
                        }
                        for _, __, fwd_result, rev_result in model_results
                    ]
                )

    diffs_df = polars.DataFrame(asymmetry)
    print(diffs_df)

    for model, model_diffs_df in diffs_df.group_by("model"):
        fig = go.Figure()
        fig.add_trace(
            go.Violin(
                x=model_diffs_df["dataset"],
                y=model_diffs_df["diff"],
                name="\u0394",
                points=False,
                box_visible=False,
                meanline_visible=False,
            )
        )
        fig.update_layout(
            template="simple_white",
            xaxis_title="Dataset Name",
            yaxis_title="\u0394=matcher(a, b)-matcher(b, a)",
        )
        fig.show()
