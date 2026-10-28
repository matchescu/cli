import pytest

from matchescu.cli.commands.ambiguity.selection import SeparabilityBounds


def test_max_separability_mirrors_max_auc():
    assert SeparabilityBounds(0.2, 0.9).max_separability == pytest.approx(0.8)


@pytest.mark.parametrize(
    "separability, expected",
    [
        pytest.param(0.1, False, id="below-min"),
        pytest.param(0.2, True, id="at-min"),
        pytest.param(0.5, True, id="inside"),
        pytest.param(0.8, True, id="at-max"),
        pytest.param(0.81, False, id="above-max"),
        pytest.param(None, False, id="unknown"),
    ],
)
def test_contains_is_inclusive_band(separability, expected):
    assert SeparabilityBounds(0.2, 0.9).contains(separability) is expected


@pytest.mark.parametrize(
    "min_separability, max_auc",
    [
        pytest.param(-0.1, 0.9, id="negative-min"),
        pytest.param(0.2, 0.5, id="max-auc-at-chance"),
        pytest.param(0.2, 1.1, id="max-auc-above-one"),
        pytest.param(0.7, 0.8, id="empty-band"),
    ],
)
def test_invalid_bounds_raise(min_separability, max_auc):
    with pytest.raises(ValueError, match="separability|max_auc"):
        SeparabilityBounds(min_separability, max_auc)


def test_default_bounds():
    assert SeparabilityBounds() == SeparabilityBounds(0.2, 0.9)
