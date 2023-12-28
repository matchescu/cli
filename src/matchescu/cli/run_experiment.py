import datetime
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
import plotly.graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator

from matchescu.cli._compute_quality_metrics import compute_metrics, ModelType
from matchescu.cli._entity_resolution import match_entities
from matchescu.cli._experiment_setups import MiniBuy, ExistingData
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


def _load_result(output_dir: Path, threshold: float) -> dict:
    log = get_logger("load-result")
    results_path = _experiment_result_file_name(output_dir, threshold)
    log.info("loading results from %s", results_path)
    with open(results_path, "r") as f:
        results = orjson.loads(f.read())
    log.info("loaded results from %s", results_path)
    return results


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
def run_experiment(
    experiments: list[ExperimentType], show_graph: bool, perform_matching: bool
) -> None:
    log = get_logger()
    process_count = os.cpu_count() - 1
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

    for setup, ground_truth in setups.items():
        for model_type in [ModelType.FSM, ModelType.ALG]:
            df = pd.DataFrame()
            for x in range(0, 100, 1):
                t = x / 100
                result_future = pool.submit(_load_result, setup.output_directory, t)
                future = pool.submit(
                    compute_metrics, ground_truth, result_future.result(), model_type
                )
                row = future.result()
                df = pd.concat([df, pd.DataFrame(row, index=[t])])
            df.to_csv(_experiment_metrics(setup.output_directory, model_type), sep=";")
            log.info("saved %s metrics", model_type)
        log.info("metrics computed")

    if not show_graph:
        return

    log.info("showing graph")
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
