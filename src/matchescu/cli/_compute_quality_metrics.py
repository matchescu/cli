import sys

import click
import json

from matchescu.metrics.fsm import extract_fsm_result_model, precision, recall, f1


INPUT_FILE = click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True, readable=True)


@click.command("compute-metrics")
@click.option("-g", "--gold-standard", type=INPUT_FILE, required=True)
@click.option("-r", "--entity-resolution-results", type=INPUT_FILE, required=True)
@click.option("-t", "--model-type", type=click.Choice(["fsm", "serf", "algebraic"]), required=True, default="fsm")
def compute_metrics(gold_standard: str, entity_resolution_results: str, model_type: str):
    with open(gold_standard, "r") as gold_standard_json:
        ground_truth_obj = json.load(gold_standard_json)
    if model_type not in ground_truth_obj:
        print("The", model_type, "model is not supported by the ground truth", file=sys.stderr)
        sys.exit(1)
    truth = extract_fsm_result_model(ground_truth_obj[model_type])

    with open(entity_resolution_results, "r") as err_json:
        result_obj = json.load(err_json)
    if model_type not in result_obj:
        print("The", model_type, "model is not supported by the entity resolution task", file=sys.stderr)
        sys.exit(1)
    result = extract_fsm_result_model(result_obj["fsm"])

    print("precision:", precision(truth, result))
    print("recall:", recall(truth, result))
    print("f1:", f1(truth, result))
