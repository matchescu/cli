from pathlib import Path

from matchescu.matching.config import (
    AnyDatasetConfig,
    MagellanBenchmarkDataConfig,
    CsvBenchmarkDataConfig,
)
from matchescu.matching.evaluation.data.benchmark import (
    MagellanBenchmarkDataBuilder,
    CsvBenchmarkDataBuilder,
)


def new_benchmark_data_factory(params: AnyDatasetConfig, data_dir: Path):
    match params:
        case MagellanBenchmarkDataConfig():
            return MagellanBenchmarkDataBuilder(params, data_dir)
        case CsvBenchmarkDataConfig():
            return CsvBenchmarkDataBuilder(params, data_dir)
        case _:
            raise ValueError(f"Unsupported dataset params type: {type(params)}")
