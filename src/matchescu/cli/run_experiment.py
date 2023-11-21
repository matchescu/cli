import json
import logging
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any

import click
import pandas as pd
import plotly.express as px
from plotly.validators.scatter.marker import SymbolValidator

from matchescu.adt.entity_resolution_result import EntityResolutionResult
from matchescu.cli._compute_quality_metrics import compute_metrics, ModelType
from matchescu.cli._entity_resolution import match_entities
from matchescu.cli._generate import generate
from matchescu.common.partitioning import compute_partition

repo_parent_dir = Path(__file__).parent.parent.parent.parent.parent
data_dir = repo_parent_dir / "data"
abt_file = data_dir / "abt-buy" / "Abt.csv"
buy_file = data_dir / "abt-buy" / "Buy.csv"
ideal_mapping_file = data_dir / "abt-buy" / "abt_buy_perfectMapping.csv"
gt_file = data_dir / "abt-buy" / "gt.json"
er_file = data_dir / "abt-buy" / "er.json"

generator_input_file = data_dir / "Buy.csv"
output_directory = data_dir
gold_standard = str((output_directory / "Buy-ground-truth.json").absolute())
output_file = str((output_directory / "Buy-result.json").absolute())


def _generate_abt_buy_ground_truth():
    abt = pd.read_csv(abt_file, header=0, index_col="id", encoding_errors="ignore")
    buy = pd.read_csv(buy_file, header=0, index_col="id", encoding_errors="ignore")
    mapping = pd.read_csv(ideal_mapping_file, header=0)
    pair_list = []
    input_set = {}
    for index, link in mapping.iterrows():
        id_abt = link["idAbt"]
        id_buy = link["idBuy"]
        abt_ref = tuple(map(str, (*abt.loc[id_abt], id_abt)))
        buy_ref = tuple(map(str, (*buy.loc[id_buy], id_buy)))
        pair = (abt_ref, buy_ref)
        pair_list.append(pair)
        input_set[abt_ref] = None
        input_set[buy_ref] = None

    gt = EntityResolutionResult()
    gt.fsm = pair_list
    gt.algebraic = compute_partition(list(input_set), pair_list)
    return {"fsm": gt.fsm, "algebraic": gt.algebraic}


def _get_abt_buy():
    return [str(x.absolute()) for x in [abt_file, buy_file]]


def _generate_miniature_ground_truth():
    generate(
        str(generator_input_file.absolute()),
        str(output_directory.absolute()),
        gold_standard,
        [
            "name,manufacturer,price,id",
            "description,name,id",
        ],
    )

    with open(gold_standard) as f:
        return json.load(f)


def _get_mini_dataset():
    return [str((data_dir / f"{idx:05}-sub-Buy.csv").absolute()) for idx in range(1, 3)]


class ExperimentType(Enum):
    Mini = 0
    Full = 1


experiment_config = {
    ExperimentType.Mini: (_generate_miniature_ground_truth, _get_mini_dataset),
    ExperimentType.Full: (_generate_abt_buy_ground_truth, _get_abt_buy),
}


@click.command(name="run-experiment")
@click.option(
    "-t", "--experiment-type", type=click.Choice(ExperimentType), default=ExperimentType.Mini
)
def run_experiment(experiment_type: ExperimentType):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    pool = ThreadPoolExecutor(32)
    gen_ground_truth, gen_dataset = experiment_config[experiment_type]
    ground_truth = gen_ground_truth()
    input_files = gen_dataset()

    results: dict[float, dict[ModelType, Any]] = {
        threshold: result
        for threshold, result in pool.map(
            partial(match_entities, input_file=input_files),
            (x / 100 for x in range(0, 100, 1)),
        )
    }

    metrics = {}
    for model_type in [ModelType.FSM, ModelType.ALG]:
        df = pd.DataFrame()
        futures = {
            t: pool.submit(compute_metrics, ground_truth, asdict(result), model_type)
            for t, result in results.items()
        }
        for t, future in futures.items():
            row = future.result()
            df = pd.concat([df, pd.DataFrame(row, index=[t])])

        fig = px.scatter(
            df[::3],
            labels={
                "index": "Jaccard Threshold (t)",
                "value": "Measurement",
                "variable": f"{model_type} Evaluator",
            },
        )
        tick_size = 20
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
            showlegend=True,
        )
        fig.show()


if __name__ == "__main__":
    run_experiment()
