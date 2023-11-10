import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator

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
        fig = px.scatter(
            df[::5],
            labels={
                "index": "Jaccard Threshold (t)",
                "value": "Measurement",
                "variable": f"{model_type} Evaluator",
            }
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
            showlegend=True
        )
        fig.show()
