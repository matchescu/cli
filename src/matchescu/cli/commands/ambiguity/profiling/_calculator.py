from typing import Any, Protocol

from ._context import ProfileContext


class ProfileCalculator(Protocol):
    """Compute one group of profile columns from a profile context."""

    def __call__(self, context: ProfileContext) -> dict[str, Any]:
        """Return the columns of this group, in a stable order.

        Columns that cannot be computed for the dataset are ``None``, so
        every profile has the same schema.
        """
        ...
