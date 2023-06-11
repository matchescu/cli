from typing import Any, Iterable

import click
import json

from entity_resolution_measurements import *
from abstractions.data_structures import Clustering


INPUT_FILE = click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True, readable=True)


def _get_cells(row: Iterable[Iterable[Any]]) -> str:
    return "</td><td>".join(
        f"<code>({','.join(map(str, values))})</code>"
        for values in row
    )


def _build_row(
    err_row: Iterable[Iterable[Any]],
    gs_row: Iterable[Iterable[Any]],
    score: float,
    format_str: str
) -> str:
    html_row_template = f"<tr><td>{{}}</td><td>{{}}</td><td>{format_str}</td></tr>"
    result = html_row_template.format(_get_cells(err_row), _get_cells(gs_row), score)
    return result


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

    with open("out/results.html", "w") as results_html:
        results_html.write("<html><body>")
        bmd = basic_merge_distance(er_results, gold_standard)
        results_html.write(
            f"<h1>ER Quality Evaluation</h1><h2>Merge distance</h2><p>Total merge distance: {bmd}</p>"
        )

        results_html.write("<h2>Pairwise F1 Score</h2><table><thead>")
        count = len(gold_standard.feature_info)
        results_html.write(f"<tr><th colspan={count}>Entity Resolution Clustering</th>")
        results_html.write(f"<th colspan={count}>Gold Standard</th><th>Score</th></tr></thead><tbody>")
        for i, score in enumerate(pairwise_f1(er_results, gold_standard)):
            results_html.write(
                _build_row(er_results.clustered_rows[i], gold_standard.clustered_rows[i], score, "{:.2%}")
            )
        results_html.write("</tbody></table>")

        results_html.write("<h2>Variation of Information for each clustering</h2><table><thead>")
        results_html.write(f"<tr><th colspan={count}>Entity Resolution Clustering</th>")
        results_html.write(f"<th colspan={count}>Gold Standard</th><th>Score</th></tr></thead><tbody>")
        for i, score in enumerate(pairwise_f1(er_results, gold_standard)):
            results_html.write(
                _build_row(er_results.clustered_rows[i], gold_standard.clustered_rows[i], score, "{:.2f}")
            )
        results_html.write("</tbody></table>")
