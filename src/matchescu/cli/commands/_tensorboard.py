from pathlib import Path

import click
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from rich.progress import MofNCompleteColumn, Progress, TimeElapsedColumn
from tensorboard.backend.event_processing import event_accumulator

from matchescu.cli._cmd_group import matchescu
from matchescu.cli.config import EvaluationConfig, JSONConfig
from matchescu.cli.runtime import get_options, make_absolute_path


class PlotTensorboard(click.Command):
    """Read Tensorboard logs and create plots."""

    def __init__(self):
        super().__init__(
            name="plot-tensorboard",
            help="Plot values from tensorboard logs.",
            params=[
                click.Option(
                    ["-O", "--output-dir"],
                    help="output directory for results",
                    type=click.Path(file_okay=False, writable=True),
                    default="./output",
                ),
                click.Option(
                    ["-c", "--config-file"],
                    required=True,
                    type=click.Path(
                        exists=True,
                        file_okay=True,
                        dir_okay=False,
                        readable=True,
                        resolve_path=True,
                    ),
                    help="path to configuration file",
                ),
            ],
        )
        self._prog = Progress(
            *Progress.get_default_columns(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        )
        self._overall_prog = self._prog.add_task(description="overall progress")

    def _update_progress(self, description=None, advance=None):
        self._prog.update(self._overall_prog, description=description, advance=advance)

    def read_tensorboard_logs(
        self, log_dir: Path, model_name: str, dataset_name: str, features: list[str]
    ) -> pl.DataFrame:
        log_dir = Path(log_dir)
        event_files = sorted(log_dir.rglob("events.out.tfevents.*"))

        if not event_files:
            raise FileNotFoundError(f"No tfevents files found in {log_dir}")

        # epoch (step) -> {tag: value}
        records: dict = {}
        for ev_file in event_files:
            feature = None
            ff = [x for x in features if x in ev_file.parent.stem]
            if ev_file.parent != log_dir and not ff:
                continue
            if ff:
                feature = ff[0]
            ea = event_accumulator.EventAccumulator(
                str(ev_file),
                size_guidance={event_accumulator.SCALARS: 0},
            )
            ea.Reload()
            available = set(ea.Tags().get("scalars", []))
            for tag in available:
                if tag != model_name:
                    continue
                for scalar in ea.Scalars(tag):
                    step = scalar.step
                    if feature is None:
                        continue
                    records.setdefault(int(step), {})[feature] = scalar.value

        # Empty logs → empty frame with the correct schema
        if not records:
            schema = {
                "model_name": pl.String,
                "dataset_name": pl.String,
                "epoch": pl.Int64,
            }
            for tag in features:
                schema[tag] = pl.Float64
            return pl.DataFrame(schema)

        # Build wide table
        steps = sorted(records.keys())
        data = {
            "model_name": [model_name] * len(steps),
            "dataset_name": [dataset_name] * len(steps),
            "epoch": steps,
        }
        for tag in features:
            data[tag] = [records[s].get(tag) for s in steps]

        df = pl.DataFrame(data)
        # Ensure numeric columns are Float64 even when a tag is completely absent
        for tag in features:
            if tag in df.columns:
                df = df.with_columns(pl.col(tag).cast(pl.Float64, strict=False))
        return df

    def plot_tensorboard_metrics(
        self,
        df: pl.DataFrame,
        epoch_col: str,
        line_cols: list[str] | dict[str, str],
        smooth_span: int | None = None,
        title: str | None = None,
    ) -> go.Figure:
        col_names = list(line_cols)
        projection = [epoch_col] + col_names
        missing = [c for c in projection if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in DataFrame: {missing}")

        plot_df = df.select(projection).sort(epoch_col)

        if plot_df.is_empty():
            return go.Figure().update_layout(title=title or "No data")

        # Smoothing: trailing rolling mean so the curve stays causally aligned
        if smooth_span is not None and smooth_span > 1:
            exprs = [pl.col(epoch_col)]
            for c in line_cols:
                exprs.append(
                    pl.col(c)
                    .rolling_mean(window_size=smooth_span, min_periods=1)
                    .alias(c)
                )
            plot_df = plot_df.select(exprs)

        # High-contrast, colorblind-friendly palette (Wong, 2011)
        colors = [
            "#000000",
            "#E69F00",
            "#56B4E9",
            "#009E73",
            "#F0E442",
            "#0072B2",
            "#D55E00",
            "#CC79A7",
        ]

        fig = px.line(
            plot_df,
            x=epoch_col,
            y=col_names,
            labels=line_cols if isinstance(line_cols, dict) else None,
            title=title,
            color_discrete_sequence=colors,
        )

        # Academic black-on-white styling
        fig.update_layout(
            template="simple_white",
            font={"family": "Arial, sans-serif", "size": 14, "color": "black"},
            title_font={"size": 16, "color": "black"},
            legend={
                "title_text": "",
                "bgcolor": "rgba(255,255,255,0)",
                "bordercolor": "black",
                "borderwidth": 1,
            },
            xaxis={
                "showgrid": True,
                "gridcolor": "lightgrey",
                "linecolor": "black",
                "linewidth": 1,
                "ticks": "outside",
            },
            yaxis={
                "showgrid": True,
                "gridcolor": "lightgrey",
                "linecolor": "black",
                "linewidth": 1,
                "ticks": "outside",
            },
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        fig.update_traces(line={"width": 2})
        return fig

    def invoke(self, ctx: click.Context):
        """Execute the asymmetry evaluation command."""
        root_dir = get_options(ctx).root_dir
        cfg: EvaluationConfig = (
            JSONConfig(
                make_absolute_path(ctx.params["config_file"], root_dir),
                EvaluationConfig,
            )
            .load()
            .config_obj
        )
        output_dir = make_absolute_path(ctx.params["output_dir"], root_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._prog.update(
            self._overall_prog, total=len(cfg.matching) * len(cfg.benchmark_data)
        )

        for ds_config in cfg.benchmark_data:
            ds_name = Path(ds_config.dataset.directory).stem
            for model_config in cfg.matching:
                tb_dir = (
                    make_absolute_path(
                        str(model_config.path).format(dataset_name=ds_name), root_dir
                    ).parent
                    / "tensorboard"
                )
                training_df = self.read_tensorboard_logs(
                    tb_dir,
                    model_config.name,
                    ds_name,
                    ["average_loss", "dev_f1", "test_f1", "dev_mcc", "test_mcc"],
                )
                figure = self.plot_tensorboard_metrics(
                    training_df,
                    "epoch",
                    {
                        "average_loss": "Avg Loss",
                        "dev_f1": "F1 score",
                        "dev_mcc": "Matthews' Correlation Coefficient",
                    },
                    3,
                )
                output_path = (
                    output_dir / f"fig_{model_config.name}_{ds_name}_training.png"
                )
                figure.write_image(output_path, scale=4)


matchescu.add_command(PlotTensorboard())
