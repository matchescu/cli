import click
import pytest
from click.testing import CliRunner

from matchescu.cli.commands.ambiguity._shared import (
    parse_split_ratio,
    with_profile_options,
    with_selection_settings,
)
from matchescu.cli.commands.ambiguity.profiling import ProfileOptions, SplitRatio
from matchescu.cli.commands.ambiguity.selection import (
    SelectionSettings,
    SeparabilityBounds,
)

USAGE_ERROR = 2


def test_parse_split_ratio_returns_split_ratio():
    assert parse_split_ratio(None, None, "8,1,1") == SplitRatio((8.0, 1.0, 1.0))


def test_parse_split_ratio_reports_invalid_ratio_as_bad_parameter():
    with pytest.raises(click.BadParameter, match="split ratio"):
        parse_split_ratio(None, None, "3,0,1")


@pytest.fixture
def options_command():
    received = {}

    @click.command()
    @click.option("--name")
    @with_profile_options
    def command(name, profile_options):
        received.update(name=name, options=profile_options)

    return command, received


def test_profile_options_default_to_core_profile(options_command):
    command, received = options_command

    CliRunner().invoke(command, ["--name", "x"], catch_exceptions=False)

    assert received == {"name": "x", "options": ProfileOptions()}


def test_profile_options_collect_flags_into_profile_options(options_command):
    command, received = options_command
    args = [
        "--negative-similarity", "--neg-pos-ratio", "2.5", "--surface-baseline",
        "--synthetic-detectability", "--seed", "7", "--cv-folds", "3",
        "--split-ratio", "8,1,1",
    ]  # fmt: skip

    CliRunner().invoke(command, args, catch_exceptions=False)

    assert received["options"] == ProfileOptions(
        split_ratio=SplitRatio((8.0, 1.0, 1.0)),
        negative_similarity=True,
        neg_pos_ratio=2.5,
        surface_baseline=True,
        synthetic_detectability=True,
        seed=7,
        cv_folds=3,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(["--cv-folds", "1"], id="one-fold"),
        pytest.param(["--neg-pos-ratio", "0"], id="zero-neg-pos-ratio"),
    ],
)
def test_profile_options_reject_out_of_range_values(options_command, args):
    command, _ = options_command

    result = CliRunner().invoke(command, args)

    assert result.exit_code == USAGE_ERROR


@pytest.fixture
def selection_command():
    received = {}

    @click.command()
    @with_selection_settings
    def command(selection_settings):
        received.update(settings=selection_settings)

    return command, received


def test_selection_settings_default(selection_command):
    command, received = selection_command

    CliRunner().invoke(command, [], catch_exceptions=False)

    assert received["settings"] == SelectionSettings()


def test_selection_settings_collect_options(selection_command):
    command, received = selection_command
    args = [
        "--min-separability", "0.3", "--max-auc", "0.85",
        "--alpha-criterion", "gmm-balance", "--target-resolvable", "0.4",
    ]  # fmt: skip

    CliRunner().invoke(command, args, catch_exceptions=False)

    assert received["settings"] == SelectionSettings(
        SeparabilityBounds(0.3, 0.85), "gmm-balance", 0.4
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(["--alpha-criterion", "largest"], id="unknown-criterion"),
        pytest.param(["--max-auc", "0.5"], id="max-auc-at-chance"),
        pytest.param(
            ["--min-separability", "0.7", "--max-auc", "0.8"], id="empty-band"
        ),
        pytest.param(["--target-resolvable", "1.5"], id="target-above-one"),
    ],
)
def test_selection_settings_reject_invalid_values(selection_command, args):
    command, _ = selection_command

    result = CliRunner().invoke(command, args)

    assert result.exit_code == USAGE_ERROR
