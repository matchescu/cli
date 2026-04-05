from typing import Literal, Annotated, Union

from pydantic import Field

from matchescu.comparison_space.persistence import CsvComparisonSpaceFileParams
from matchescu.matching.config import ConfigModel, AnyDatasetConfig


class ModelConfig(ConfigModel):
    name: str
    path: str
    tokenizer: str


class DittoModelConfig(ModelConfig):
    type: Literal["ditto"] = "ditto"
    lhs_columns: list[str] | None = None
    rhs_columns: list[str] | None = None


class DeepMatcherModelConfig(ModelConfig):
    type: Literal["deepmatcher"] = "deepmatcher"
    tokenizer:str = "google-bert/bert-base-uncased"
    attribute_map: dict[str, str] | None = None
    attribute_vector_length: int = 30
    excluded_attributes: list[str | int] | None = None


AnyModelConfig = Annotated[
    Union[DittoModelConfig, DeepMatcherModelConfig],
    Field(discriminator="type"),
]


class ComparisonSpaceConfig(ConfigModel):
    file_name: str
    sample_count: int
    neg_pos_ratio: float = 8.0
    match_bridge_ratio: float = 3.5

    has_header: bool = True
    left_id_col: str | int = 0,
    left_source_col: str | int | None = None
    right_id_col: str | int = 1
    right_source_col: str | int | None = None

    def to_csv_params(self, source_fallback: str|None = None) -> CsvComparisonSpaceFileParams:
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


class EvaluationConfig(ConfigModel):
    benchmark_data: list[EvaluationBenchmarkConfig]
    models: list[AnyModelConfig]
