import sys
from enum import Enum
from typing import Iterable

import click
import json

from matchescu.metrics.algebraic import extract_algebraic_result_model, twi, rand_index, adjusted_rand_index, \
    pair_precision, pair_recall, pair_comparison_measure, cluster_precision, cluster_recall, cluster_comparison_measure
from matchescu.metrics.fsm import extract_fsm_result_model, precision, recall, f1

INPUT_FILE = click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True, readable=True)


class ModelType(str, Enum):
    FSM = "fsm"
    SERF = "serf"
    ALG = "algebraic"


def _compute_fsm_metrics(
        ground_truth_obj: dict[str, Iterable[Iterable[Iterable]]],
        result_obj: dict[str, Iterable[Iterable[Iterable]]],
):
    truth = extract_fsm_result_model(ground_truth_obj[ModelType.FSM])
    result = extract_fsm_result_model(result_obj[ModelType.FSM])

    print("precision:", precision(truth, result))
    print("recall:", recall(truth, result))
    print("f1:", f1(truth, result))


def _compute_algebraic_metrics(
        ground_truth_obj: dict[str, Iterable[Iterable[Iterable]]],
        result_obj: dict[str, Iterable[Iterable[Iterable]]],
):
    truth = extract_algebraic_result_model(ground_truth_obj[ModelType.ALG])
    result = extract_algebraic_result_model(result_obj[ModelType.ALG])

    print("twi:", twi(truth, result))
    print("rand index:", rand_index(truth, result))
    print("adjusted rand index:", adjusted_rand_index(truth, result))
    print()
    print("pairwise precision:", pair_precision(truth, result))
    print("pairwise recall:", pair_recall(truth, result))
    print("pairwise comparison measure:", pair_comparison_measure(truth, result))
    print()
    print("cluster precision:", cluster_precision(truth, result))
    print("cluster recall:", cluster_recall(truth, result))
    print("cluster comparison measure:", cluster_comparison_measure(truth, result))


METRICS = {
    ModelType.FSM: _compute_fsm_metrics,
    ModelType.ALG: _compute_algebraic_metrics
}


@click.command("compute-metrics")
@click.option("-g", "--gold-standard", type=INPUT_FILE, required=True)
@click.option("-r", "--entity-resolution-results", type=INPUT_FILE, required=True)
@click.option("-t", "--model-type", type=click.Choice(ModelType), required=True, default="fsm")
def compute_metrics(gold_standard: str, entity_resolution_results: str, model_type: ModelType):
    with open(gold_standard, "r") as gold_standard_json:
        ground_truth_obj = json.load(gold_standard_json)
    if model_type not in ground_truth_obj:
        print("The", model_type, "model is not supported by the ground truth", file=sys.stderr)
        sys.exit(1)
    with open(entity_resolution_results, "r") as err_json:
        result_obj = json.load(err_json)
    if model_type not in result_obj:
        print("The", model_type, "model is not supported by the entity resolution task", file=sys.stderr)
        sys.exit(1)
    if model_type not in METRICS:
        print("The", model_type, "model is not supported by the app", file=sys.stderr)
        sys.exit(1)

    METRICS[model_type](ground_truth_obj, result_obj)
