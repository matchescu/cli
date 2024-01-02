import datetime
import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from enum import StrEnum
from functools import partial
from pathlib import Path
from time import time

import click
import orjson
import pandas as pd
import plotly.express as px
from plotly.validators.scatter.marker import SymbolValidator

from matchescu.cli._compute_quality_metrics import compute_metrics, ModelType
from matchescu.cli._entity_resolution import match_entities
from matchescu.cli._experiment_setups import MiniBuy, ExistingData
from matchescu.instrumentation.timer import timer
from matchescu.logs import get_logger

repo_parent_dir = Path(__file__).parent.parent.parent.parent.parent
data_dir = repo_parent_dir / "data"


class ExperimentType(StrEnum):
    Mini = "mini"
    AbtBuy = "abt-buy"
    AmazonGoogleProducts = "amazon-google"
    DBLP_ACM = "dblp-acm"
    DBLP_Scholar = "dblp-scholar"


experiment_config = {
    ExperimentType.Mini: partial(MiniBuy, data_dir=data_dir),
    ExperimentType.AbtBuy: partial(
        ExistingData,
        data_dir=data_dir / "abt-buy",
        ds1_name="Abt.csv",
        ds2_name="Buy.csv",
        perfect_mapping_name="abt_buy_perfectMapping.csv",
        ds1_pm_id_col="idAbt",
        ds2_pm_id_col="idBuy",
    ),
    ExperimentType.AmazonGoogleProducts: partial(
        ExistingData,
        data_dir=data_dir / "Amazon-GoogleProducts",
        ds1_name="Amazon.csv",
        ds2_name="GoogleProducts.csv",
        perfect_mapping_name="Amzon_GoogleProducts_perfectMapping.csv",
        ds1_pm_id_col="idAmazon",
        ds2_pm_id_col="idGoogleBase",
    ),
    ExperimentType.DBLP_ACM: partial(
        ExistingData,
        data_dir=data_dir / "DBLP-ACM",
        ds1_name="DBLP2.csv",
        ds2_name="ACM.csv",
        perfect_mapping_name="DBLP-ACM_perfectMapping.csv",
        ds1_pm_id_col="idDBLP",
        ds2_pm_id_col="idACM",
    ),
    ExperimentType.DBLP_Scholar: partial(
        ExistingData,
        data_dir=data_dir / "DBLP-Scholar",
        ds1_name="DBLP1.csv",
        ds2_name="Scholar.csv",
        perfect_mapping_name="DBLP-Scholar_perfectMapping.csv",
        ds1_pm_id_col="idDBLP",
        ds2_pm_id_col="idScholar",
    ),
}


def _results_dir(output_dir: Path) -> Path:
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def _experiment_result_file_name(output_dir: Path, threshold: float) -> str:
    return str((_results_dir(output_dir) / f"{threshold}.json").absolute())


def _experiment_metrics(output_dir: Path, model_type: ModelType) -> str:
    return str((output_dir / f"{model_type.value.lower()}-metrics.csv").absolute())


class compute_threshold_metrics:
    def __init__(self, model_types: list[ModelType], results_dir: Path):
        self.__model_types = model_types
        self.__results_dir = results_dir

    @staticmethod
    @timer("load-er-result")
    def _load_result(output_dir: Path, threshold: float) -> dict:
        results_path = _experiment_result_file_name(output_dir, threshold)
        with open(results_path, "r") as f:
            results = orjson.loads(f.read())
        return results

    @timer("compute-threshold-metrics")
    def __call__(
        self, call_args: tuple[float, dict]
    ) -> tuple[float, dict[ModelType, dict[str, float]]]:
        threshold, ground_truth = call_args
        log = get_logger("compute-metrics")
        er_result = self._load_result(self.__results_dir, threshold)
        threshold_metrics = {}
        for model_type in self.__model_types:
            log.info("computing [%s]@t=%.2f", model_type, threshold)
            threshold_metrics[model_type] = compute_metrics(
                ground_truth, er_result, model_type, log
            )
        return threshold, threshold_metrics


@click.command(name="run-experiment")
@click.option(
    "-e",
    "--experiment",
    "experiments",
    type=click.Choice(ExperimentType),
    default=[ExperimentType.Mini],
    multiple=True,
)
@click.option("-g", "--show-graph", type=click.BOOL, default=True)
@click.option("-m", "--perform-matching", type=click.BOOL, default=True)
@click.option(
    "-d",
    "--image-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True),
    required=True,
)
@click.option("-c", "--calculate-metrics", type=click.BOOL, default=True)
def run_experiment(
    experiments: list[ExperimentType],
    show_graph: bool,
    perform_matching: bool,
    image_dir: str,
    calculate_metrics: bool,
) -> None:
    log = get_logger()
    process_count = os.cpu_count() // 2
    model_types = [ModelType.FSM, ModelType.ALG]

    log.info("using %d parallel processes for entity resolution", process_count)

    pool = ProcessPoolExecutor(process_count)
    setups = {
        experiment_config[experiment](prepare_matching=perform_matching): {}
        for experiment in experiments
    }
    for setup in setups:
        setups[setup] = setup.generate_ground_truth()

    if perform_matching:
        start_time = time()
        log.info("performing matching using %d parallel processes", process_count)
        for setup in setups:
            log.info("performing %s entity resolution", setup)
            input_files = setup.list_dataset_files()
            for threshold, result in pool.map(
                partial(match_entities, input_file=input_files),
                (x / 100 for x in range(0, 100, 1)),
            ):
                results_path = _experiment_result_file_name(
                    setup.output_directory, threshold
                )
                log.info("saving results to %s", results_path)
                with open(results_path, "w") as f:
                    f.write(orjson.dumps(result).decode("utf-8"))
                log.info("results saved to %s", results_path)
        log.info(
            "completed matching in %s", datetime.timedelta(seconds=time() - start_time)
        )

    if calculate_metrics:
        for setup, ground_truth in setups.items():
            metrics: dict[ModelType, pd.DataFrame] = {
                mtype: pd.DataFrame() for mtype in model_types
            }
            computer = compute_threshold_metrics(model_types, setup.output_directory)
            metric_args = zip(
                (x / 100 for x in range(0, 100, 1)),
                itertools.repeat(ground_truth, 100),
            )
            for t, threshold_metrics in pool.map(computer, metric_args):
                for model_type in threshold_metrics:
                    model_type_df = metrics[model_type]
                    threshold_df = threshold_metrics[model_type]
                    model_type_df = pd.concat(
                        [model_type_df, pd.DataFrame(threshold_df, index=[t])]
                    )
                    metrics[model_type] = model_type_df
            for model_type in metrics:
                metrics[model_type].to_csv(
                    _experiment_metrics(setup.output_directory, model_type), sep=";"
                )
                log.info("saved %s metrics", model_type)

    if not show_graph:
        return

    log.info("showing graph")
    graph_groups = {ModelType.FSM: [""], ModelType.ALG: ["pairwise", "cluster", ""]}
    for setup in setups:
        for model_type in model_types:
            model_type_df = pd.read_csv(
                _experiment_metrics(setup.output_directory, model_type),
                sep=";",
                header=0,
                index_col=0,
            )
            visited = set()
            for group in graph_groups[model_type]:
                graph_df = model_type_df[
                    [
                        col
                        for col in model_type_df
                        if col.startswith(group)
                        and not any(col.startswith(v) for v in visited)
                    ]
                ]
                visited.add(group)
                fig = px.line(
                    graph_df,
                    labels={
                        "index": "Jaccard Threshold (t)",
                        "value": "Measurement",
                        "variable": " ",
                    },
                )
                symbols = [
                    s
                    for s in SymbolValidator().values[2::3]
                    if len(s) < 3 or not (s[-3:] == "dot" or s[-3:] == "pen")
                ]
                for trace, symbol in zip(fig.data, symbols):
                    trace.update(
                        mode="lines+markers", marker_symbol=symbol, marker_size=8
                    )
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
                    showlegend=True,
                    legend=dict(
                        orientation="h", yanchor="top", y=-0.125, xanchor="center", x=0.49
                    ),
                )
                file_name = f"{setup}-{model_type.value.lower()}-{group or 'main'}.png"
                file_path = os.path.join(image_dir, file_name)
                fig.write_image(file_path)
                fig.show()


if __name__ == "__main__":
    run_experiment()
