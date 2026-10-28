import pytest

from matchescu.cli.commands.ambiguity.profiling import ProfileOptions


@pytest.mark.parametrize(
    "neg_pos_ratio",
    [pytest.param(0.0, id="zero"), pytest.param(-1.0, id="negative")],
)
def test_neg_pos_ratio_not_positive_raises(neg_pos_ratio):
    with pytest.raises(ValueError, match="neg_pos_ratio"):
        ProfileOptions(neg_pos_ratio=neg_pos_ratio)


def test_cv_folds_below_two_raises():
    with pytest.raises(ValueError, match="cv_folds"):
        ProfileOptions(cv_folds=1)
