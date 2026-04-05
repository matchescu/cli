from matchescu.matching.config import (
    AnyDatasetConfig,
    MagellanBenchmarkDataConfig,
    CsvBenchmarkDataConfig,
)
from matchescu.matching.evaluation.data.benchmark import (
    MagellanBenchmarkDataFactory,
    CsvBenchmarkDataFactory,
)


def new_benchmark_data_factory(params: AnyDatasetConfig):
    match params:
        case MagellanBenchmarkDataConfig():
            return MagellanBenchmarkDataFactory(params)
        case CsvBenchmarkDataConfig():
            return CsvBenchmarkDataFactory(params)
        case _:
            raise ValueError(f"Unsupported dataset params type: {type(params)}")
