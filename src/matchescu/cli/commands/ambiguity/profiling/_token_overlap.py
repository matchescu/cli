from dataclasses import dataclass


@dataclass(frozen=True)
class TokenOverlap:
    """Token overlap of an ordered pair of texts, left then right.

    The full features keep the order of the pair; the symmetric features
    are invariant to swapping left and right. A model that only gains from
    the full features is reading direction from surface asymmetry.

    :ivar left_size: distinct tokens of the left text.
    :ivar right_size: distinct tokens of the right text.
    :ivar shared: distinct tokens of both texts.
    """

    left_size: int
    right_size: int
    shared: int

    @classmethod
    def of(cls, left: frozenset[str], right: frozenset[str]) -> "TokenOverlap":
        """Return the overlap of two token sets."""
        return cls(len(left), len(right), len(left & right))

    @property
    def left_coverage(self) -> float:
        """Share of the left tokens found on the right (0 for no tokens)."""
        return self.shared / self.left_size if self.left_size else 0.0

    @property
    def right_coverage(self) -> float:
        """Share of the right tokens found on the left (0 for no tokens)."""
        return self.shared / self.right_size if self.right_size else 0.0

    def full(self) -> tuple[float, ...]:
        """Order-aware features: coverages, their gap, size gap, emptiness."""
        return (
            self.left_coverage,
            self.right_coverage,
            self.left_coverage - self.right_coverage,
            self.left_size - self.right_size,
            float(self.left_size == 0),
            float(self.right_size == 0),
        )

    def symmetric(self) -> tuple[float, ...]:
        """Swap-invariant features: coverage bounds and gaps, empty count."""
        coverages = (self.left_coverage, self.right_coverage)
        return (
            min(coverages),
            max(coverages),
            abs(self.left_coverage - self.right_coverage),
            abs(self.left_size - self.right_size),
            float(self.left_size == 0) + float(self.right_size == 0),
        )
