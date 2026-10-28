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


def _row(alpha, separability, pi=None, boundary=BAYES, gap=_ABSENT, detect=_ABSENT):
    row = {
        "alpha": alpha,
        "content_separability": separability,
        "margin_pi_resolvable": pi,
        "margin_boundary_type": boundary,
    }
    if gap is not _ABSENT:
        row["surface_auc_gap"] = gap
    if detect is not _ABSENT:
        row["detect_auc_lr"] = detect
    return row


def _choose(rows, criterion="smallest", target_resolvable=0.5, max_detect_auc=0.7):
    settings = SelectionSettings(
        SeparabilityBounds(0.2, 0.9), criterion, target_resolvable, max_detect_auc
    )
    return AlphaSelector(settings).choose(rows)


def test_registry_lists_criteria():
    assert sorted(ALPHA_CRITERIA) == [
        "gmm-balance",
        "order-headroom",
        "smallest",
        "surface-gap",
    ]


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


def test_order_headroom_picks_largest_gap_among_undetectable_alphas():
    rows = [
        _row(0.5, 0.3, gap=0.01, detect=0.6),
        _row(0.65, 0.4, gap=0.08, detect=0.6),
        _row(0.8, 0.5, gap=0.02, detect=0.6),
    ]

    assert _choose(rows, "order-headroom").alpha == 0.65


def test_order_headroom_excludes_alphas_above_detectability_cap():
    rows = [_row(0.5, 0.3, gap=0.01, detect=0.6), _row(0.8, 0.5, gap=0.2, detect=0.8)]

    assert _choose(rows, "order-headroom").alpha == 0.5


def test_order_headroom_keeps_alpha_at_detectability_cap():
    rows = [_row(0.8, 0.5, gap=0.2, detect=0.7)]

    assert _choose(rows, "order-headroom").alpha == 0.8


def test_order_headroom_uses_given_detectability_cap():
    rows = [_row(0.5, 0.3, gap=0.01, detect=0.6), _row(0.8, 0.5, gap=0.2, detect=0.8)]

    assert _choose(rows, "order-headroom", max_detect_auc=0.85).alpha == 0.8


def test_order_headroom_breaks_ties_towards_smaller_alpha():
    rows = [_row(0.8, 0.5, gap=0.05, detect=0.6), _row(0.65, 0.4, gap=0.05, detect=0.6)]

    assert _choose(rows, "order-headroom").alpha == 0.65


@pytest.mark.parametrize(
    "detect",
    [
        pytest.param(0.9, id="detectable"),
        pytest.param(None, id="skipped"),
        pytest.param(_ABSENT, id="not-computed"),
    ],
)
def test_order_headroom_without_undetectable_alpha_is_reported(detect):
    rows = [_row(0.5, 0.3, gap=0.1, detect=detect)]

    assert _choose(rows, "order-headroom") == AlphaChoice(
        None, AlphaChoice.NO_UNDETECTABLE_ALPHA, 1
    )


def test_unknown_criterion_raises():
    with pytest.raises(ValueError, match="criterion"):
        SelectionSettings(SeparabilityBounds(), "largest", 0.5)


@pytest.mark.parametrize("target", [-0.1, 1.1])
def test_target_resolvable_outside_unit_interval_raises(target):
    with pytest.raises(ValueError, match="target_resolvable"):
        SelectionSettings(SeparabilityBounds(), "gmm-balance", target)


@pytest.mark.parametrize("cap", [0.4, 1.1])
def test_max_detect_auc_outside_auc_range_raises(cap):
    with pytest.raises(ValueError, match="max_detect_auc"):
        SelectionSettings(SeparabilityBounds(), "order-headroom", 0.5, cap)


def test_surface_gap_requires_surface_baseline():
    selector = AlphaSelector(SelectionSettings(SeparabilityBounds(), "surface-gap"))

    with pytest.raises(ValueError, match="surface_baseline"):
        selector.check_requirements(ProfileOptions())


@pytest.mark.parametrize(
    "options, missing",
    [
        pytest.param(
            ProfileOptions(surface_baseline=True),
            "synthetic_detectability",
            id="no-detectability",
        ),
        pytest.param(
            ProfileOptions(synthetic_detectability=True),
            "surface_baseline",
            id="no-surface-baseline",
        ),
    ],
)
def test_order_headroom_requires_both_baselines(options, missing):
    selector = AlphaSelector(SelectionSettings(SeparabilityBounds(), "order-headroom"))

    with pytest.raises(ValueError, match=missing):
        selector.check_requirements(options)


@pytest.mark.parametrize(
    "criterion, options",
    [
        pytest.param("smallest", ProfileOptions(), id="smallest"),
        pytest.param("gmm-balance", ProfileOptions(), id="gmm-balance"),
        pytest.param(
            "surface-gap", ProfileOptions(surface_baseline=True), id="surface-gap"
        ),
        pytest.param(
            "order-headroom",
            ProfileOptions(surface_baseline=True, synthetic_detectability=True),
            id="order-headroom",
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
    settings = SelectionSettings(SeparabilityBounds(0.3, 0.85), "gmm-balance", 0.4, 0.8)

    assert settings.as_row() == {
        "alpha_criterion": "gmm-balance",
        "min_separability": 0.3,
        "max_auc": 0.85,
        "target_resolvable": 0.4,
        "max_detect_auc": 0.8,
    }
