from pathlib import Path

from matchescu.cli._generate import generate
from matchescu.cli._entity_resolution import match_entities
from matchescu.cli._compute_quality_metrics import compute_metrics, ModelType

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
    for threshold in range(0, 100, 10):
        t = threshold / 100
        match_entities(input_files, t)
        compute_metrics(gold_standard, output_file, ModelType.FSM)
