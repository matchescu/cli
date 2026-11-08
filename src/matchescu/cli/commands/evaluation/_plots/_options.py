from collections.abc import Callable
from typing import Any

import click

DEFAULT_PLOT_DPI = 120


def plot_dpi_option(**kwargs: Any) -> Callable:
    defaults = {
        "help": "resolution of the generated plots; ignored without a plot directory",
        "default": DEFAULT_PLOT_DPI,
        "show_default": True,
    }
    return click.option(
        "--plot-dpi", type=click.IntRange(min=1), **{**defaults, **kwargs}
    )
