from typing import Annotated, Literal

from matchescu.comparison_space.persistence import CsvComparisonSpaceFileParams
from matchescu.matching.config import AnyDatasetConfig, ConfigModel
from pydantic import Field


class ModelConfig(ConfigModel):
    name: str
    path: str
    tokenizer: str | None = None


class DittoConfig(ModelConfig):
    type: Literal["ditto", "multiclass"] = "ditto"
    lhs_columns: list[str] | None = None
    rhs_columns: list[str] | None = None


class DeepMatcherConfig(ModelConfig):
    type: Literal["deepmatcher"] = "deepmatcher"
    tokenizer: str = "google-bert/bert-base-uncased"
    attribute_map: dict[str, str] | None = None
    attribute_vector_length: int = 30
    excluded_attributes: list[str | int] | None = None


class DeepERConfig(ModelConfig):
    type: Literal["deeper"] = "deeper"
    tokenizer: str = "google-bert/bert-base-uncased"
    attribute_map: dict[str, str] | None = None
    attribute_vector_length: int = 30
    excluded_attributes: list[str | int] | None = None


AnyMatcherConfig = Annotated[
    DittoConfig | DeepMatcherConfig | DeepERConfig,
    Field(discriminator="type"),
]


class ComparisonSpaceConfig(ConfigModel):
    file_name: str
    sample_count: int | None = None
    neg_pos_ratio: float = 4.0
    match_bridge_ratio: float = 2.0

    has_header: bool = True
    left_id_col: str | int = (0,)
    left_source_col: str | int | None = None
    right_id_col: str | int = 1
    right_source_col: str | int | None = None

    excluded_files: list[str] | None = None

    def to_csv_params(
        self, source_fallback: str | None = None
    ) -> CsvComparisonSpaceFileParams:
        return CsvComparisonSpaceFileParams(
            self.has_header,
            source_fallback,
            self.left_id_col,
            self.left_source_col,
            self.right_id_col,
            self.right_source_col,
        )


class EvaluationBenchmarkConfig(ConfigModel):
    dataset: AnyDatasetConfig
    comparison_space: ComparisonSpaceConfig
    clusters_file_name: str | None = None


class ClusteringConfig(ConfigModel):
    algorithms: list[str]


class EvaluationConfig(ConfigModel):
    benchmark_data: list[EvaluationBenchmarkConfig]
    matching: list[AnyMatcherConfig] | None = None
    clustering: ClusteringConfig | None = None


class AmbiguityDataConfig(ConfigModel):
    dataset: AnyDatasetConfig
    ambiguity_targets: list[str] = None
    min_ambiguity: float = None


class AmbiguityConfig(ConfigModel):
    input_data: list[AmbiguityDataConfig]
