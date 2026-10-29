from enum import StrEnum


class PairingScope(StrEnum):
    """Describes how to choose pairs based on top-k ambiguous neighbors.

    :ivar ANY: keep the pair (a, b) when either a or b consider the other one
        a top neighbor.
    :ivar MUTUAL: keep a pair (a, b) only when both a and b consider the other
        one a top neighbor.
    """

    ANY = "any"
    MUTUAL = "mutual"
