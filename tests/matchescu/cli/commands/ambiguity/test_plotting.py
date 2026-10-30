import pytest

from matchescu.cli.commands.ambiguity._alpha_sweep import AlphaSweepResult
from matchescu.cli.commands.ambiguity._plotting import render_alpha_sweep_plot
from matchescu.cli.commands.ambiguity.selection import AlphaChoice, SeparabilityBounds


def _row(alpha, separability, pi, gap=None, detect=None):
    return {
        "alpha": alpha,
        "content_separability": separability,
        "margin_pi_resolvable": pi,
        "surface_auc_gap": gap,
        "detect_auc_lr": detect,
    }


@pytest.mark.parametrize(
    "result, criterion",
    [
        pytest.param(
            AlphaSweepResult(
                (_row(0.5, 0.1, 0.2, 0.01, 0.6), _row(0.65, 0.4, 0.5, 0.08, 0.65)),
                {"smallest": AlphaChoice(0.65, AlphaChoice.SELECTED, 1)},
            ),
            "smallest",
            id="selected",
        ),
        pytest.param(
            AlphaSweepResult(
                (_row(0.5, None, None), _row(0.65, 0.9, None)),
                {"smallest": AlphaChoice(None, AlphaChoice.NO_FEASIBLE_ALPHA, 0)},
            ),
            "smallest",
            id="missing-values-no-choice",
        ),
        pytest.param(
            AlphaSweepResult(
                (_row(0.5, 0.3, 0.4, 0.05, 0.7),),
                {"gmm-balance": AlphaChoice(0.5, AlphaChoice.SELECTED, 1)},
            ),
            "gmm-balance",
            id="per-criterion-filename",
        ),
        pytest.param(
            AlphaSweepResult((_row(0.5, 0.3, 0.4),), {}),
            None,
            id="no-criterion-no-alpha-line",
        ),
    ],
)
def test_alpha_sweep_plot_is_written(result, criterion, tmp_path):
    name = (
        "alpha_sweep_ds.png" if criterion is None else f"alpha_sweep_ds_{criterion}.png"
    )
    out_path = tmp_path / name

    render_alpha_sweep_plot(result, SeparabilityBounds(), out_path, criterion)

    assert out_path.stat().st_size > 0
