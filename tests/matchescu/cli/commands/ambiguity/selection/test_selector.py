import pytest

from matchescu.cli.commands.ambiguity.profiling import ProfileOptions
from matchescu.cli.commands.ambiguity.selection import (
    ALPHA_CRITERIA,
    AlphaChoice,
    AlphaSelector,
    SelectionSettings,
    SeparabilityBounds,
)

BAYES = "bayes_optimal"
MIDPOINT = "mean_midpoint"
_ABSENT = object()


def _row(alpha, separability, pi=None, boundary=BAYES, gap=_ABSENT):
    row = {
        "alpha": alpha,
        "content_separability": separability,
        "margin_pi_resolvable": pi,
        "margin_boundary_type": boundary,
    }
    if gap is not _ABSENT:
        row["surface_auc_gap"] = gap
    return row


def _choose(rows, criterion="smallest", target_resolvable=0.5):
    settings = SelectionSettings(
        SeparabilityBounds(0.2, 0.9), criterion, target_resolvable
    )
    return AlphaSelector(settings).choose(rows)


def test_registry_lists_criteria():
    assert sorted(ALPHA_CRITERIA) == ["gmm-balance", "smallest", "surface-gap"]


def test_smallest_picks_smallest_feasible_alpha():
    rows = [_row(0.5, 0.1), _row(0.65, 0.3), _row(0.8, 0.5), _row(0.95, 0.9)]

    assert _choose(rows) == AlphaChoice(0.65, AlphaChoice.SELECTED, 2)


def test_no_feasible_alpha_is_reported():
    rows = [_row(0.5, 0.1), _row(0.95, 0.9)]

    assert _choose(rows) == AlphaChoice(None, AlphaChoice.NO_FEASIBLE_ALPHA, 0)


def test_gmm_balance_picks_pi_closest_to_target():
    rows = [_row(0.5, 0.3, pi=0.2), _row(0.65, 0.4, pi=0.45), _row(0.8, 0.5, pi=0.7)]

    assert _choose(rows, "gmm-balance").alpha == 0.65


def test_gmm_balance_uses_given_target():
    rows = [_row(0.5, 0.3, pi=0.2), _row(0.65, 0.4, pi=0.45), _row(0.8, 0.5, pi=0.7)]

    assert _choose(rows, "gmm-balance", target_resolvable=0.75).alpha == 0.8


def test_gmm_balance_breaks_ties_towards_smaller_alpha():
    rows = [_row(0.8, 0.5, pi=0.6), _row(0.65, 0.4, pi=0.4)]

    assert _choose(rows, "gmm-balance").alpha == 0.65


def test_gmm_balance_ignores_degenerate_fits():
    rows = [_row(0.5, 0.3, pi=0.5, boundary=MIDPOINT), _row(0.8, 0.5, pi=0.9)]

    assert _choose(rows, "gmm-balance").alpha == 0.8


def test_gmm_balance_without_bayes_optimal_fit_is_reported():
    rows = [_row(0.5, 0.3, pi=0.5, boundary=MIDPOINT), _row(0.8, 0.5)]

    assert _choose(rows, "gmm-balance") == AlphaChoice(
        None, AlphaChoice.NO_BAYES_OPTIMAL_FIT, 2
    )


def test_surface_gap_picks_largest_gap():
    rows = [
        _row(0.5, 0.3, gap=0.01),
        _row(0.65, 0.4, gap=0.08),
        _row(0.8, 0.5, gap=0.02),
    ]

    assert _choose(rows, "surface-gap").alpha == 0.65


def test_surface_gap_skips_infeasible_alphas():
    rows = [_row(0.5, 0.1, gap=0.5), _row(0.8, 0.5, gap=0.02)]

    assert _choose(rows, "surface-gap").alpha == 0.8


@pytest.mark.parametrize(
    "gap",
    [pytest.param(_ABSENT, id="not-computed"), pytest.param(None, id="skipped")],
)
def test_surface_gap_without_gap_values_is_reported(gap):
    rows = [_row(0.5, 0.3, gap=gap)]

    assert _choose(rows, "surface-gap").reason == AlphaChoice.NO_SURFACE_BASELINE


def test_unknown_criterion_raises():
    with pytest.raises(ValueError, match="criterion"):
        SelectionSettings(SeparabilityBounds(), "largest", 0.5)


@pytest.mark.parametrize("target", [-0.1, 1.1])
def test_target_resolvable_outside_unit_interval_raises(target):
    with pytest.raises(ValueError, match="target_resolvable"):
        SelectionSettings(SeparabilityBounds(), "gmm-balance", target)


def test_surface_gap_requires_surface_baseline():
    selector = AlphaSelector(SelectionSettings(SeparabilityBounds(), "surface-gap"))

    with pytest.raises(ValueError, match="surface_baseline"):
        selector.check_requirements(ProfileOptions())


@pytest.mark.parametrize(
    "criterion, options",
    [
        pytest.param("smallest", ProfileOptions(), id="smallest"),
        pytest.param("gmm-balance", ProfileOptions(), id="gmm-balance"),
        pytest.param(
            "surface-gap", ProfileOptions(surface_baseline=True), id="surface-gap"
        ),
    ],
)
def test_met_requirements_pass(criterion, options):
    selector = AlphaSelector(SelectionSettings(SeparabilityBounds(), criterion))

    selector.check_requirements(options)


@pytest.mark.parametrize(
    "separability, expected",
    [pytest.param(0.3, True, id="inside"), pytest.param(0.95, False, id="outside")],
)
def test_is_feasible_applies_bounds_to_row(separability, expected):
    selector = AlphaSelector(SelectionSettings(SeparabilityBounds(), "smallest"))

    assert selector.is_feasible(_row(0.5, separability)) is expected


def test_settings_row_reports_choice_parameters():
    settings = SelectionSettings(SeparabilityBounds(0.3, 0.85), "gmm-balance", 0.4)

    assert settings.as_row() == {
        "alpha_criterion": "gmm-balance",
        "min_separability": 0.3,
        "max_auc": 0.85,
        "target_resolvable": 0.4,
    }
