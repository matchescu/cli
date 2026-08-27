import logging
import os
from pathlib import Path
from typing import Iterable

import click
import polars
from rich.progress import Progress, TimeElapsedColumn, MofNCompleteColumn

from matchescu.cli.config import new_benchmark_data_factory, EvaluationConfig
from matchescu.cli.data import load_comparison_space
from matchescu.cli.models import new_matcher
from matchescu.cli.runtime import get_options, make_absolute_path
from matchescu.matching.evaluation.data.benchmark import BenchmarkData
from matchescu.reference_store.comparison_space import BinaryComparisonSpace
from matchescu.typing import EntityReference
from ._cmd_group import evaluate, EvalOptions
from ...config._config import AnyMatcherConfig

os.environ["DISABLE_TQDM"] = "true"


logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_parallel_utils").setLevel(logging.ERROR)


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
                )
            ],
        )
        self._prog = Progress(
            *Progress.get_default_columns(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        )
        self._overall_prog = self._prog.add_task(description="overall progress")
        self._dataset_prog = self._prog.add_task(description="dataset")
        self._model_prog = self._prog.add_task(description="model")

    def _load_data(
        self, root_dir: Path, cfg: EvaluationConfig
    ) -> Iterable[tuple[BenchmarkData, BinaryComparisonSpace]]:
        for ds_config in cfg.benchmark_data:
            self._prog.update(
                self._overall_prog, description=f"loading {ds_config.dataset.directory}"
            )
            builder = new_benchmark_data_factory(ds_config.dataset, root_dir)
            benchmark_data = builder.load_data().create()
            ds_dir = root_dir / ds_config.dataset.directory
            cs = load_comparison_space(
                benchmark_data, ds_dir, ds_config.comparison_space
            )
            self._prog.update(
                self._overall_prog, description=f"{ds_config.dataset.directory} loaded"
            )
            yield benchmark_data, cs

    def _run_matcher(
        self,
        root_dir: Path,
        benchmark_data: BenchmarkData,
        cs_refs: list[EntityReference],
        model_config: AnyMatcherConfig,
    ) -> Iterable[dict]:
        matcher = new_matcher(model_config, root_dir, benchmark_data.name)
        model_results = []
        for x, y in cs_refs:
            xy_result = matcher(x, y)
            yx_result = matcher(y, x)
            yield {
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
            self._prog.advance(self._model_prog)
            self._prog.advance(self._dataset_prog)
            self._prog.advance(self._overall_prog)
        return model_results

    def invoke(self, ctx: click.Context):
        """Execute the asymmetry evaluation command."""
        eval_opts: EvalOptions[EvaluationConfig] = get_options(ctx)
        root_dir = eval_opts.root_dir
        cfg = eval_opts.config
        output_dir = make_absolute_path(ctx.params["output_dir"], root_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._prog.start()

        self._prog.start_task(self._overall_prog)
        data = list(self._load_data(root_dir, cfg))
        n_models = len(cfg.matching)
        n_total_ops = n_models * sum(len(cs) for _, cs in data)
        self._prog.update(self._overall_prog, total=n_total_ops, completed=0)
        dataframes = []
        for benchmark_data, cs in data:
            self._prog.stop_task(self._dataset_prog)
            cs_refs: list[EntityReference] = list(
                map(benchmark_data.id_table.get_all, cs)
            )
            n_ds_ops = len(cs) * len(cfg.matching)
            self._prog.update(
                self._dataset_prog,
                description=benchmark_data.name,
                total=n_ds_ops,
                completed=0,
            )
            self._prog.start_task(self._dataset_prog)

            for model_config in cfg.matching:
                self._prog.stop_task(self._model_prog)
                self._prog.update(
                    self._model_prog,
                    description=model_config.name,
                    total=len(cs),
                    completed=0,
                )
                self._prog.start_task(self._model_prog)
                matcher_df = polars.DataFrame(
                    list(
                        self._run_matcher(
                            root_dir, benchmark_data, cs_refs, model_config
                        )
                    )
                )
                file_name = f"{benchmark_data.name}-{model_config.name}-asymmetry.csv"
                matcher_df.write_csv(output_dir / file_name, include_header=True)
                dataframes.append(matcher_df)
        concatenated = polars.concat(dataframes, how="vertical", strict=True)
        concatenated.write_csv(output_dir / "all.csv", include_header=True)


# Register the command with the evaluate group
evaluate.add_command(AsymmetryCommand())
