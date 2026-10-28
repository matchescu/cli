from os import PathLike
from pathlib import Path

import click
from matchescu.matching.evaluation.data.benchmark import CsvBenchmarkData

from matchescu.cli.config import JSONConfig, new_benchmark_data_factory
from matchescu.cli.runtime import make_absolute_path

from ...config._config import AmbiguityConfig, AmbiguityDataConfig
from .profiling import SplitRatio


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
    :raises pydantic.ValidationError: when the JSON does not match the
        ambiguity configuration schema.
    """
    return (
        JSONConfig(make_absolute_path(config_file, root_dir), AmbiguityConfig)
        .load()
        .config_obj
    )


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


def parse_secondary_features(raw: str | None) -> list[str] | None:
    """Split a comma-separated ``--secondary-features`` value into names.

    A ``None`` result is meaningful, not an error: it tells the caller the
    option was omitted or passed empty, so per-dataset configuration should
    apply instead of forcing "no secondary features".

    :param raw: comma-separated feature names as typed on the command line.
    :returns: stripped feature names in argument order; ``None`` when ``raw``
        is ``None`` or empty.
    """
    if not raw:
        return None
    return [feature.strip() for feature in raw.split(",") if feature.strip()]


def parse_alphas(raw: str) -> tuple[float, ...]:
    """Parse comma-separated alpha weights for the harmonic-mean pick.

    :param raw: comma-separated weights as typed on the command line; entries
        must lie in (0, 1) for generation — out-of-range values surface as a
        `ValueError` from the ambiguity generator constructor, which owns that
        validation.
    :returns: weights in argument order.
    """
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


def parse_split_ratio(
    _ctx: click.Context, _param: click.Parameter, raw: str
) -> SplitRatio:
    """Click callback turning ``--split-ratio`` into a :class:`SplitRatio`.

    :param raw: comma-separated relative split sizes, e.g. ``"3,1,1"``.
    :returns: the parsed ratio.
    :raises click.BadParameter: when a part is not a positive number.
    """
    try:
        return SplitRatio.parse(raw)
    except ValueError as err:
        raise click.BadParameter(str(err)) from err
