from collections.abc import Iterator
from contextlib import contextmanager

import click

from matchescu.cli.data import StaleReferenceGraphError

NOTHING_RECOMPUTED = "Nothing was recomputed or overwritten."
REBUILD_FROM_MATCHING = (
    "Re-run `evaluate matching` without --skip-existing to rebuild the "
    "reference graphs."
)
REBUILD_WITHOUT_SKIP = "Re-run without --skip-existing to rebuild the reference graphs."


def stale_message(reason: str, hint: str) -> str:
    return f"{reason}\n{NOTHING_RECOMPUTED} {hint}"


@contextmanager
def abort_on_stale_graph(hint: str) -> Iterator[None]:
    """Turn a stale reference graph into a friendly CLI error carrying ``hint``."""
    try:
        yield
    except StaleReferenceGraphError as error:
        raise click.ClickException(stale_message(str(error), hint)) from error
