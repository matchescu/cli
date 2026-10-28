import pytest

from matchescu.cli.commands.ambiguity._alpha_sweep import AlphaSweepResult
from matchescu.cli.commands.ambiguity._plotting import render_alpha_sweep_plot
from matchescu.cli.commands.ambiguity.selection import AlphaChoice, SeparabilityBounds


def _row(alpha, separability, pi):
    return {
        "alpha": alpha,
        "content_separability": separability,
        "margin_pi_resolvable": pi,
    }


@pytest.mark.parametrize(
    "result",
    [
        pytest.param(
            AlphaSweepResult(
                (_row(0.5, 0.1, 0.2), _row(0.65, 0.4, 0.5)),
                AlphaChoice(0.65, AlphaChoice.SELECTED, 1),
            ),
            id="selected",
        ),
        pytest.param(
            AlphaSweepResult(
                (_row(0.5, None, None), _row(0.65, 0.9, None)),
                AlphaChoice(None, AlphaChoice.NO_FEASIBLE_ALPHA, 0),
            ),
            id="missing-values-no-choice",
        ),
    ],
)
def test_alpha_sweep_plot_is_written(result, tmp_path):
    out_path = tmp_path / "alpha_sweep_ds.png"

    render_alpha_sweep_plot(result, SeparabilityBounds(), out_path)

    assert out_path.stat().st_size > 0
