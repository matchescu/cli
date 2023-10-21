import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import plotly.express as px

from matchescu.cli._compute_quality_metrics import compute_metrics, ModelType
from matchescu.cli._entity_resolution import match_entities
from matchescu.cli._generate import generate

repo_parent_dir = Path(__file__).parent.parent.parent.parent.parent
data_dir = repo_parent_dir / "data"
input_file = data_dir / "Buy.csv"
output_directory = data_dir
gold_standard = str((output_directory / "Buy-ground-truth.json").absolute())
output_file = str((output_directory / "Buy-result.json").absolute())

if __name__ == "__main__":
    generate(
        str(input_file.absolute()),
        str(output_directory.absolute()),
        gold_standard,
        [
            "name,manufacturer",
            "description,name,price,id",
        ],
    )
    input_files = [
        str((data_dir / f"{idx:05}-sub-Buy.csv").absolute()) for idx in range(1, 3)
    ]
    with open(gold_standard) as f:
        ground_truth = json.load(f)
    for model_type in ModelType:
        df = pd.DataFrame()
        for threshold in range(0, 100):
            t = threshold / 100
            result = match_entities(input_files, t)
            row = compute_metrics(ground_truth, asdict(result), model_type)
            df = pd.concat([df, pd.DataFrame(row, index=[t])])
        fig = px.line(
            df,
            labels={
                "index": "Jaccard Threshold (t)",
                "value": "Measurement",
                "variable": f"{model_type} Evaluator",
            },
        )
        fig.show()
