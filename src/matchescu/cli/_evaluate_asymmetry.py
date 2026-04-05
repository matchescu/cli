import logging
import os
from os import PathLike
from pathlib import Path

import click
import polars
import plotly.graph_objects as go
from rich.progress import Progress

from transformers import AutoTokenizer
from matchescu.reference_store.comparison_space import BinaryComparisonSpace
from matchescu.comparison_space.persistence import CsvPersistence, CsvComparisonSpaceFileParams
from matchescu.matching.matchers import DeepMatcherSimilarity, DittoSimilarity
from matchescu.matching.evaluation.data.benchmark import BenchmarkData
from matchescu.matching.evaluation.data.generation import GroundTruthComparisonSpaceGenerator
from pyresolvemetrics import precision, recall, f1

from .config import EvaluationConfig, new_benchmark_data_factory, JSONConfig, DeepMatcherModelConfig, DittoModelConfig
from ._cmd_group import matchescu
from .config._config import ComparisonSpaceConfig

os.environ["DISABLE_TQDM"] = "true"

logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_parallel_utils").setLevel(logging.ERROR)

def _new_deepmatcher(config: DeepMatcherModelConfig, root_dir: Path, dataset_name: str) -> DeepMatcherSimilarity:
    file_path = Path(str(config.path).format(dataset_name=dataset_name))
    if not file_path.is_absolute():
        file_path = root_dir / file_path
    return DeepMatcherSimilarity(
        AutoTokenizer.from_pretrained(config.tokenizer),
        config.attribute_map,
        config.attribute_vector_length,
        config.excluded_attributes
    ).load_from_file(file_path)

def _new_ditto(config: DittoModelConfig, root_dir: Path, dataset_name: str) -> DittoSimilarity:
    file_path = Path(str(config.path).format(dataset_name=dataset_name))
    if not file_path.is_absolute():
        file_path = root_dir / file_path
    return DittoSimilarity(
        AutoTokenizer.from_pretrained(config.tokenizer),
        left_cols=config.lhs_columns,
        right_cols=config.rhs_columns,
    ).load_from_file(file_path)


SIMILARITY_TYPE_MAP = {
    DeepMatcherModelConfig: _new_deepmatcher,
    DittoModelConfig: _new_ditto,
}


def _load_comparison_space(data_dir: Path, config: ComparisonSpaceConfig, data: BenchmarkData) -> BinaryComparisonSpace:
    file_path = data_dir / config.file_name
    if file_path.exists():
        persistence = CsvPersistence(file_path)
        return persistence.read(config.to_csv_params(data.name))
    else:
        csg = GroundTruthComparisonSpaceGenerator(
            data.id_table,
            data.true_matches,
            data.compute_clusters(),
            config.neg_pos_ratio,
            config.match_bridge_ratio,
            config.sample_count,
            save_comparisons=True,
            save_clusters=True
        )
        cluster_file_path = file_path.parent / f"{file_path.stem}-clusters.csv"
        return csg(file_path, cluster_file_path)


@matchescu.command()
@click.option(
    "-d",
    "--root-dir",
    default=Path.cwd(),
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True),
    help="Everything in the config files is relative to this directory. Required.",
)
@click.option(
    "-f",
    "--config-file",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True),
    help="configuration file for the evaluation pipeline"
)
def evaluate_asymmetry(root_dir: Path, config_file: str | PathLike):
    root_dir = Path(root_dir).absolute()
    config_file = Path(config_file).absolute()
    cfg = JSONConfig(config_file, EvaluationConfig).load()
    stats = []
    asymmetry = []
    with Progress() as progress:
        for ds_config in cfg.config_obj.benchmark_data:
            ds_dir = root_dir / ds_config.dataset.directory
            factory = new_benchmark_data_factory(ds_config.dataset)
            benchmark_data = factory.create(root_dir)
            cs = _load_comparison_space(ds_dir, ds_config.comparison_space, benchmark_data)
            cs_refs = list(map(benchmark_data.id_table.get_all, cs))
            cs_true = set(cmp for cmp in cs if cmp in benchmark_data.true_matches)
            total_comparisons = len(cs) * len(cfg.config_obj.models)
            ds_task = progress.add_task(ds_config.dataset.directory, total=total_comparisons)

            for model_config in cfg.config_obj.models:
                model_task = progress.add_task(model_config.name, total=len(cs))
                factory = SIMILARITY_TYPE_MAP[type(model_config)]
                similarity = factory(model_config, root_dir, benchmark_data.name)
                model_results = []
                for x, y in cs_refs:
                    model_results.append(
                        (x.id, y.id, similarity(x, y), similarity(y, x))
                    )
                    progress.update(ds_task, advance=1)
                    progress.update(model_task, advance=1)

                asymmetry.extend([
                    {
                        "dataset": benchmark_data.name,
                        "model": model_config.name,
                        "diff": fwd_result.label_weights[fwd_result.label] - rev_result.label_weights[rev_result.label],
                        "abs_diff": abs(fwd_result.label_weights[fwd_result.label] - rev_result.label_weights[rev_result.label])
                    }
                    for _, __ , fwd_result, rev_result in model_results
                ])
                fwd_matches = set((x, y) for x, y, fwd, _ in model_results if fwd.label > 0)
                rev_matches = set((x, y) for x, y, _, rev in model_results if rev.label > 0)
                stats.append({
                    "fwd_precision": precision(cs_true, fwd_matches),
                    "fwd_recall": recall(cs_true, fwd_matches),
                    "fwd_f1": f1(cs_true, fwd_matches),
                    "rev_precision": precision(cs_true, rev_matches),
                    "rev_recall": recall(cs_true, rev_matches),
                    "rev_f1": f1(cs_true, rev_matches),
                    "dataset": benchmark_data.name,
                    "model": model_config.name
                })
    stats_df = polars.DataFrame(stats)
    print(stats_df)
    diffs_df = polars.DataFrame(asymmetry)
    print(diffs_df)

    for model, model_diffs_df in diffs_df.group_by("model"):
        fig = go.Figure()
        fig.add_trace(
            go.Violin(
                x=model_diffs_df["dataset"],
                y=model_diffs_df["diff"],
                name="\u0394",
                points=False,
                box_visible=False,
                meanline_visible=False,
            )
        )
        fig.update_layout(
            template="simple_white",
            xaxis_title="Dataset Name",
            yaxis_title="\u0394=matcher(a, b)-matcher(b, a)",
        )
        fig.show()
