from dataclasses import asdict

import click
import json


@click.command("transform-err")
@click.option(
    "-i",
    "--input-file",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, resolve_path=True, readable=True
    ),
    required=True,
)
@click.option(
    "-o",
    "--output-file",
    type=click.Path(file_okay=True, dir_okay=False, resolve_path=True, writable=True),
    required=True,
)
def transform_results(input_file: str, output_file: str):
    with open(input_file, "r") as json_file:
        json_obj = json.load(json_file)

    from entity_resolution_results.ppjoin import transform_result

    clustering = transform_result(json_obj)
    with open(output_file, "w") as cluster_file:
        json.dump(asdict(clustering), cluster_file, indent=4)
