from pathlib import Path

import click
import polars
from matchescu.typing import EntityReference
from rich.progress import Progress

from matchescu.cli.config import (
    AnyMatcherConfig,
    EvaluationBenchmarkConfig,
    EvaluationConfig,
    new_benchmark_data_factory,
)
from matchescu.cli.data import count_comparison_space_rows, load_comparison_space
from matchescu.cli.models import new_matcher
from matchescu.cli.runtime import get_options, make_absolute_path

from ._eval_opts import EvalOptions
from ._parallel import UnitReporter, UnitRunner


def _asymmetry_unit(
    ds_config: EvaluationBenchmarkConfig,
    model_config: AnyMatcherConfig,
    seed: int,
    root_dir: Path,
    reporter: UnitReporter,
) -> tuple[str, list[dict]]:
    """Compute the input-order asymmetry records for one (dataset, model) unit."""
    builder = new_benchmark_data_factory(ds_config.dataset, root_dir)
    benchmark_data = builder.load_data().create()
    ds_dir = root_dir / ds_config.dataset.directory
    cs = load_comparison_space(
        benchmark_data,
        ds_dir,
        ds_config.comparison_space,
        seed=seed,
    )
    cs_refs: list[EntityReference] = list(map(benchmark_data.id_table.get_all, cs))
    matcher = new_matcher(model_config, root_dir, benchmark_data.name)
    records = []
    for x, y in cs_refs:
        xy_result = matcher(x, y)
        yx_result = matcher(y, x)
        records.append(
            {
                "dataset": benchmark_data.name,
                "model": model_config.name,
                "left_id": x.id.label,
                "left_source": x.id.source,
                "right_id": y.id.label,
                "right_source": y.id.source,
                "xy_label": xy_result.label,
                "yx_label": yx_result.label,
                "xy_label_weight": xy_result.label_weights[xy_result.label],
                "yx_label_weight": yx_result.label_weights[yx_result.label],
                "hamming": int(abs(xy_result.label - yx_result.label)),
            }
        )
        reporter.advance(1)
    return benchmark_data.name, records


class AsymmetryCommand(click.Command):
    """Evaluate parameter input order matcher asymmetry."""

    def __init__(self):
        super().__init__(
            name="asymmetry",
            help="Evaluate parameter input order matcher asymmetry.",
            params=[
                click.Option(
                    ["-O", "--output-dir"],
                    help="output directory for results",
                    type=click.Path(file_okay=False, writable=True),
                    default="./output",
                ),
                click.Option(
                    ["-w", "--workers"],
                    help="number of worker processes (1 runs in-process)",
                    type=click.IntRange(min=1),
                    default=None,
                ),
            ],
        )

    def invoke(self, ctx: click.Context):
        """Execute the asymmetry evaluation command."""
        eval_opts: EvalOptions[EvaluationConfig] = get_options(ctx)
        root_dir = eval_opts.root_dir
        cfg = eval_opts.config
        output_dir = make_absolute_path(ctx.params["output_dir"], root_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        workers = ctx.params["workers"]

        per_ds_count = {
            ds_config.dataset.directory: count_comparison_space_rows(
                root_dir / ds_config.dataset.directory, ds_config.comparison_space
            )
            for ds_config in cfg.benchmark_data
        }
        units = [
            (
                f"{ds_config.dataset.directory}/{model_config.name}",
                (ds_config, model_config, cfg.seed, root_dir),
                per_ds_count[ds_config.dataset.directory],
            )
            for ds_config in cfg.benchmark_data
            for model_config in cfg.matching
        ]
        dataframes = []
        with Progress() as progress:
            runner = UnitRunner(workers, progress)
            for key, (ds_name, records) in runner.run(units, _asymmetry_unit):
                model_name = key.split("/")[-1]
                matcher_df = polars.DataFrame(records)
                file_name = f"{ds_name}-{model_name}-asymmetry.csv"
                matcher_df.write_csv(output_dir / file_name, include_header=True)
                dataframes.append(matcher_df)
        concatenated = polars.concat(dataframes, how="vertical", strict=True)
        concatenated.write_csv(output_dir / "all.csv", include_header=True)
