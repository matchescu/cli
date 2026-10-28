import click
import pytest

from matchescu.cli.commands.ambiguity._shared import parse_split_ratio
from matchescu.cli.commands.ambiguity.profiling import SplitRatio


def test_parse_split_ratio_returns_split_ratio():
    assert parse_split_ratio(None, None, "8,1,1") == SplitRatio((8.0, 1.0, 1.0))


def test_parse_split_ratio_reports_invalid_ratio_as_bad_parameter():
    with pytest.raises(click.BadParameter, match="split ratio"):
        parse_split_ratio(None, None, "3,0,1")
