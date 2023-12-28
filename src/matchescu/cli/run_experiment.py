import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, is_dataclass
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Any

import click
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator

from matchescu.cli._compute_quality_metrics import compute_metrics, ModelType
from matchescu.cli._entity_resolution import match_entities
from matchescu.cli._experiment_setups import MiniBuy, AbtBuy

repo_parent_dir = Path(__file__).parent.parent.parent.parent.parent
data_dir = repo_parent_dir / "data"


class ExperimentType(StrEnum):
    Mini = "mini"
    AbtBuy = "abt-buy"


experiment_config = {
    ExperimentType.Mini: MiniBuy,
    ExperimentType.AbtBuy: AbtBuy,
}


def _experiment_result_file_name(output_dir: Path) -> str:
    return str((output_dir / "results.json").absolute())


def _experiment_metrics(output_dir: Path, model_type: ModelType) -> str:
    return str((output_dir / f"{model_type.value.lower()}-metrics.csv").absolute())


class DataclassJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if is_dataclass(obj):
            return asdict(obj)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


@click.command(name="run-experiment")
@click.option(
    "-e",
    "--experiment",
    "experiments",
    type=click.Choice(ExperimentType),
    default=[ExperimentType.Mini],
    multiple=True,
)
@click.option("-g", "--show-graph", type=click.BOOL, is_flag=True, default=True)
@click.option("-m", "--perform-matching", type=click.BOOL, default=True)
def run_experiment(experiments: list[ExperimentType], show_graph: bool, perform_matching: bool) -> None:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    pool = ProcessPoolExecutor(20)
    setups = {
        experiment_config[experiment](data_dir, perform_matching): {}
        for experiment in experiments
    }
    for setup in setups:
        setups[setup] = setup.generate_ground_truth()

    if perform_matching:
        for setup in setups:
            input_files = setup.list_dataset_files()
            results: dict[float, dict[str, Any]] = {
                threshold: result
                for threshold, result in pool.map(
                    partial(match_entities, input_file=input_files),
                    (x / 100 for x in range(0, 100, 1)),
                )
            }
            with open(_experiment_result_file_name(setup.output_directory), "w") as f:
                json.dump(results, f, indent=4, cls=DataclassJSONEncoder)

    for setup, ground_truth in setups.items():
        with open(_experiment_result_file_name(setup.output_directory), "r") as f:
            results = json.load(f)
        for model_type in [ModelType.FSM, ModelType.ALG]:
            df = pd.DataFrame()
            futures = {
                t: pool.submit(compute_metrics, ground_truth, result, model_type)
                for t, result in results.items()
            }
            for t, future in futures.items():
                row = future.result()
                df = pd.concat([df, pd.DataFrame(row, index=[t])])
            df.to_csv(_experiment_metrics(setup.output_directory, model_type), sep=";")

    if not show_graph:
        return

    for setup in setups:
        for model_type in [ModelType.FSM, ModelType.ALG]:
            df = pd.read_csv(
                _experiment_metrics(setup.output_directory, model_type),
                sep=";",
                header=0,
                index_col=0,
            )
            fig = px.line(
                df,
                labels={
                    "index": "Jaccard Threshold (t)",
                    "value": "Measurement",
                    "variable": f"{model_type} Model",
                },
            )
            symbols = [
                s
                for s in SymbolValidator().values[2::3]
                if len(s) < 3 or not (s[-3:] == "dot" or s[-3:] == "pen")
            ]
            for trace, symbol in zip(fig.data, symbols):
                trace.update(mode="lines+markers", marker_symbol=symbol, marker_size=8)
            fig.update_layout(
                width=800,
                height=600,
                plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis=dict(
                    title="Jaccard Threshold (t)",
                    showline=True,
                    linecolor="black",
                    mirror=True,
                    ticks="outside",
                    tickfont=dict(size=12, color="black"),
                ),
                yaxis=dict(
                    title="Value",
                    showline=True,
                    linecolor="black",
                    mirror=True,
                    ticks="outside",
                    tickfont=dict(size=12, color="black"),
                ),
                showlegend=False,
                updatemenus=[
                    go.layout.Updatemenu(
                        type="buttons",
                        showactive=False,
                        buttons=list(
                            [
                                dict(
                                    label="Show Legend",
                                    method="relayout",
                                    args=["showlegend", True],
                                ),
                                dict(
                                    label="Hide Legend",
                                    method="relayout",
                                    args=["showlegend", False],
                                ),
                            ]
                        ),
                    )
                ],
            )
            fig.show()


if __name__ == "__main__":
    run_experiment()
