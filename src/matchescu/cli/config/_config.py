from typing import Literal, Annotated, Union

from pydantic import FilePath, Field

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


class EvaluationConfig(ConfigModel):
    benchmark_data: list[AnyDatasetConfig]
    models: list[AnyModelConfig]
