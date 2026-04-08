from pathlib import Path

from transformers import AutoTokenizer

from matchescu.cli.config import (
    DeepMatcherConfig,
    DittoConfig,
    AnyMatcherConfig,
)
from matchescu.matching.matchers import DeepMatcherSimilarity, DittoSimilarity
from matchescu.similarity import Matcher


def _new_deepmatcher(
    config: DeepMatcherConfig, file_path: Path
) -> DeepMatcherSimilarity:
    return DeepMatcherSimilarity(
        AutoTokenizer.from_pretrained(config.tokenizer),
        config.attribute_map,
        config.attribute_vector_length,
        config.excluded_attributes,
    ).load_from_file(file_path)


def _new_ditto(config: DittoConfig, file_path: Path) -> DittoSimilarity:
    return DittoSimilarity(
        AutoTokenizer.from_pretrained(config.tokenizer or config.name),
        left_cols=config.lhs_columns,
        right_cols=config.rhs_columns,
    ).load_from_file(file_path)


_CONFIG_TYPE_TO_MATCHER_MAP = {
    DeepMatcherConfig: _new_deepmatcher,
    DittoConfig: _new_ditto,
}


def new_matcher(config: AnyMatcherConfig, root_dir: Path, dataset: str) -> Matcher:
    factory = _CONFIG_TYPE_TO_MATCHER_MAP.get(config.__class__)
    if factory is None:
        raise ValueError(f"unknown model config '{config.__class__}'")
    file_path = Path(str(config.path).format(dataset_name=dataset))
    if not file_path.is_absolute():
        file_path = root_dir / file_path
    return factory(config, file_path)
