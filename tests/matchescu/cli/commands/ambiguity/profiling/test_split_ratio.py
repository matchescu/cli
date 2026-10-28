import pytest

from matchescu.cli.commands.ambiguity.profiling import SplitRatio


@pytest.mark.parametrize(
    "text, expected",
    [
        pytest.param("3,1,1", 0.2, id="default"),
        pytest.param("8, 1, 1", 0.1, id="spaces"),
        pytest.param("0.7,0.3", 0.3, id="fractions"),
    ],
)
def test_smallest_share_is_smallest_part_over_total(text, expected):
    assert SplitRatio.parse(text).smallest_share == pytest.approx(expected)


def test_parse_keeps_parts_in_order():
    assert SplitRatio.parse("3,1,2").parts == (3.0, 1.0, 2.0)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("", id="empty"),
        pytest.param("3,0,1", id="zero-part"),
        pytest.param("3,-1,1", id="negative-part"),
    ],
)
def test_parse_rejects_invalid_ratio(text):
    with pytest.raises(ValueError, match="split ratio"):
        SplitRatio.parse(text)


def test_default_ratio_is_three_one_one():
    assert SplitRatio.default() == SplitRatio.parse("3,1,1")
