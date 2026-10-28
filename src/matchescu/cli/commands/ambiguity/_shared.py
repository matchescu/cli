from os import PathLike
from pathlib import Path

import click
from matchescu.matching.evaluation.data.benchmark import CsvBenchmarkData
from pydantic import ValidationError

from matchescu.cli.config import JSONConfig, new_benchmark_data_factory
from matchescu.cli.runtime import make_absolute_path

from ._config import AmbiguityConfig, AmbiguityDataConfig

CONFIG_FILE_HINT: str = "'-c' / '--config-file'"


def load_ambiguity_config(
    config_file: str | PathLike, root_dir: Path
) -> AmbiguityConfig:
    """Load and validate the JSON configuration of the ambiguity commands.

    :param config_file: path of the configuration JSON; resolved relative to
        `root_dir` like every command option, keeping configs portable across
        invocation directories.
    :param root_dir: the CLI invocation's root directory.
    :returns: the validated ambiguity configuration.
    :raises FileNotFoundError: when `config_file` does not exist.
    :raises json.JSONDecodeError: when `config_file` contains invalid JSON.
    :raises click.BadParameter: when the JSON does not match the ambiguity
        configuration schema; the message lists every invalid setting.
    """
    try:
        return (
            JSONConfig(make_absolute_path(config_file, root_dir), AmbiguityConfig)
            .load()
            .config_obj
        )
    except ValidationError as err:
        raise click.BadParameter(str(err), param_hint=CONFIG_FILE_HINT) from err


def load_benchmark_data(
    input_cfg: AmbiguityDataConfig, root_dir: Path
) -> CsvBenchmarkData:
    """Load one configured benchmark dataset entirely into memory.

    Files are read eagerly, so id table, true matches and clusters are all
    materialised before the return and neither the generator nor the analyser
    touch the file system afterwards.

    :param input_cfg: a single `AmbiguityConfig.input_data` entry selecting
        the dataset and its file layout.
    :param root_dir: the CLI invocation's root directory; the dataset's
        relative paths resolve against it.
    :returns: the fully materialised benchmark data.
    """
    return new_benchmark_data_factory(input_cfg.dataset, root_dir).load_data().create()
