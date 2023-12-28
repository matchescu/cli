import cProfile
import json
import sys
from enum import Enum
from typing import Iterable, Any

import click

from matchescu.instrumentation.timer import timer
from matchescu.metrics.algebraic import (
    twi,
    rand_index,
    adjusted_rand_index,
    pair_precision,
    pair_recall,
    pair_comparison_measure,
    cluster_precision,
    cluster_recall,
    cluster_comparison_measure,
)
from matchescu.metrics.fsm import extract_fsm_result_model, precision, recall, f1
from matchescu.metrics.serf import (
    extract_serf_result_model,
    basic_merge_distance,
    pairwise_f1,
    variation_of_information,
    pairwise_precision,
    pairwise_recall,
)

INPUT_FILE = click.Path(
    exists=True, file_okay=True, dir_okay=False, resolve_path=True, readable=True
)


class ModelType(str, Enum):
    __DESC = {
        "fsm": "Fellegi-Sunter",
        "serf": "Stanford Entity Resolution Framework",
        "algebraic": "Algebraic",
    }
    FSM = "fsm"
    SERF = "serf"
    ALG = "algebraic"

    def __str__(self):
        return self.__DESC[self.value]


def _compute_fsm_metrics(
    ground_truth_obj: dict[str, Iterable[Iterable[Iterable]]],
    result_obj: dict[str, Iterable[Iterable[Iterable]]],
) -> dict[str, float]:
    truth = extract_fsm_result_model(ground_truth_obj[ModelType.FSM])
    result = extract_fsm_result_model(result_obj[ModelType.FSM])

    return {
        "precision": precision(truth, result),
        "recall": recall(truth, result),
        "f1": f1(truth, result),
    }


def _convert_to_algebraic(
    input_data: Iterable[Iterable[Iterable]],
) -> list[set[tuple]]:
    return [
        set(tuple(v for v in partition_item) for partition_item in partition_class)
        for partition_class in input_data
    ]


def _compute_algebraic_metrics(
    ground_truth_obj: dict[str, Iterable[Iterable[Iterable]]],
    result_obj: dict[str, Iterable[Iterable[Iterable]]],
) -> dict[str, float]:
    truth = _convert_to_algebraic(ground_truth_obj[ModelType.ALG])
    result = _convert_to_algebraic(result_obj[ModelType.ALG])

    return {
        "twi": twi(truth, result),
        "rand index": rand_index(truth, result),
        "adjusted rand index": adjusted_rand_index(truth, result),
        "pairwise precision": pair_precision(truth, result),
        "pairwise recall": pair_recall(truth, result),
        "pairwise comparison measure": pair_comparison_measure(truth, result),
        "cluster precision": cluster_precision(truth, result),
        "cluster recall": cluster_recall(truth, result),
        "cluster comparison measure": cluster_comparison_measure(truth, result),
    }


def _compute_serf_metrics(
    ground_truth_obj: dict[str, Iterable[Iterable[Iterable]]],
    result_obj: dict[str, Iterable[Iterable[Iterable]]],
) -> dict[str, float]:
    standard = extract_serf_result_model(ground_truth_obj[ModelType.SERF])
    result = extract_serf_result_model(result_obj[ModelType.SERF])

    return {
        "merge distance": basic_merge_distance(result, standard),
        "pairwise precision": pairwise_precision(result, standard),
        "pairwise recall": pairwise_recall(result, standard),
        "pairwise f1": pairwise_f1(result, standard),
        "variation of information": variation_of_information(result, standard),
    }


METRICS = {
    ModelType.FSM: _compute_fsm_metrics,
    ModelType.ALG: _compute_algebraic_metrics,
    ModelType.SERF: _compute_serf_metrics,
}


@click.command("compute-metrics")
@click.option("-g", "--gold-standard", type=INPUT_FILE, required=True)
@click.option("-r", "--entity-resolution-results", type=INPUT_FILE, required=True)
@click.option(
    "-t", "--model-type", type=click.Choice(ModelType), required=True, default="fsm"
)
def main(gold_standard: str, entity_resolution_results: str, model_type: ModelType):
    with open(gold_standard, "r") as gold_standard_json:
        ground_truth_obj = json.load(gold_standard_json)
    with open(entity_resolution_results, "r") as err_json:
        result_obj = json.load(err_json)
    for metric_name, value in compute_metrics(
        ground_truth_obj, result_obj, model_type
    ).items():
        print(f"{metric_name}:", value)


@timer("compute-metrics")
def compute_metrics(
    gold_standard: dict[str, Any],
    entity_resolution_results: dict[str, Any],
    model_type: ModelType,
) -> dict[str, float]:
    if model_type not in gold_standard:
        print(
            "The",
            model_type,
            "model is not supported by the ground truth",
            file=sys.stderr,
        )
        sys.exit(1)
    if model_type not in entity_resolution_results:
        print(
            "The",
            model_type,
            "model is not supported by the entity resolution task",
            file=sys.stderr,
        )
        sys.exit(1)
    if model_type not in METRICS:
        print("The", model_type, "model is not supported by the app", file=sys.stderr)
        sys.exit(1)

    quality_eval = METRICS[model_type]
    return quality_eval(gold_standard, entity_resolution_results)

