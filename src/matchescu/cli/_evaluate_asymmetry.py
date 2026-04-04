from os import PathLike
from pathlib import Path

import click
import polars
import plotly.graph_objects as go

from transformers import AutoTokenizer
from matchescu.comparison_space.generation._binary_csg import BinaryComparisonSpaceGenerator
from matchescu.comparison_space.generation.blocking import GroundTruthBlocker
from matchescu.matching.matchers import DeepMatcherSimilarity, DittoSimilarity
from pyresolvemetrics import precision, recall, f1

from .config import EvaluationConfig, new_benchmark_data_factory, JSONConfig, DeepMatcherModelConfig, DittoModelConfig
from ._cmd_group import matchescu


def _new_deepmatcher(config: DeepMatcherModelConfig, dataset_name: str) -> DeepMatcherSimilarity:
    file_path = str(config.path).format(dataset_name=dataset_name)
    return DeepMatcherSimilarity(
        AutoTokenizer.from_pretrained(config.tokenizer),
        config.attribute_map,
        config.attribute_vector_length,
        config.excluded_attributes
    ).load_from_file(file_path)

def _new_ditto(config: DittoModelConfig, dataset_name: str) -> DittoSimilarity:
    file_path = str(config.path).format(dataset_name=dataset_name)
    return DittoSimilarity(
        AutoTokenizer.from_pretrained(config.tokenizer),
        left_cols=config.lhs_columns,
        right_cols=config.rhs_columns,
    ).load_from_file(file_path)


SIMILARITY_TYPE_MAP = {
    DeepMatcherModelConfig: _new_deepmatcher,
    DittoModelConfig: _new_ditto,
}


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
    for ds_config in cfg.config_obj.benchmark_data:
        factory = new_benchmark_data_factory(ds_config)
        benchmark_data = factory.create(root_dir)
        blocker = GroundTruthBlocker(benchmark_data.id_table, benchmark_data.true_matches, max_count=10000)
        generate_cs = BinaryComparisonSpaceGenerator().add_blocker(blocker)
        cs = generate_cs()
        cs_refs = list(map(benchmark_data.id_table.get_all, cs))
        cs_true = set((x, y) for x, y in cs if (x, y) in benchmark_data.true_matches)

        for model_config in cfg.config_obj.models:
            factory = SIMILARITY_TYPE_MAP[type(model_config)]
            similarity = factory(model_config, benchmark_data.name)
            model_results = [
                (x.id, y.id, similarity(x, y), similarity(y, x))
                for x, y in cs_refs
            ]

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
