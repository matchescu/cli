import click
import json

from entity_resolution_measurements import *
from abstractions.data_structures import Clustering


INPUT_FILE = click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True, readable=True)


@click.command("compute-metrics")
@click.option("-g", "--gold-standard", type=INPUT_FILE, required=True)
@click.option("-r", "--entity-resolution-results", type=INPUT_FILE, required=True)
def compute_metrics(gold_standard: str, entity_resolution_results: str):
    with open(gold_standard, "r") as gold_standard_json:
        gold_standard_json_obj = json.load(gold_standard_json)
        gold_standard = Clustering.from_nested_lists(gold_standard_json_obj["clustered_rows"])

    with open(entity_resolution_results, "r") as err_json:
        err_json_obj = json.load(err_json)
        er_results = Clustering.from_nested_lists(err_json_obj["clustered_rows"])

    print("merge distance:", basic_merge_distance(er_results, gold_standard))
    print("pairwise F1 scores")
    for i, score in enumerate(pairwise_f1(er_results, gold_standard)):
        print(f"{i}: {score:.2f}%")
    print("variation of information")
    for i, score in enumerate(variation_of_information(er_results, gold_standard)):
        print(f"{i}: {score:.2f}%")
