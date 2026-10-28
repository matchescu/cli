from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class SplitRatio:
    """Relative sizes of the splits a dataset will be divided into.

    Profiles use the ratio to judge whether the largest bridge component can
    fit inside the smallest split without leaking clusters across splits.

    :ivar parts: relative split sizes, e.g. ``(3, 1, 1)`` for train, valid
        and test.
    :raises ValueError: when there are no parts or a part is not positive.
    """

    DEFAULT: ClassVar[str] = "3,1,1"
    SEPARATOR: ClassVar[str] = ","

    parts: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.parts or min(self.parts) <= 0:
            raise ValueError(f"split ratio needs positive parts, got {self.parts}")

    @classmethod
    def parse(cls, text: str) -> "SplitRatio":
        """Parse a comma-separated ratio such as ``"3,1,1"``.

        :param text: relative split sizes separated by commas.
        :returns: the parsed ratio.
        :raises ValueError: when a part is not a number or not positive.
        """
        return cls(tuple(float(p) for p in text.split(cls.SEPARATOR) if p.strip()))

    @classmethod
    def default(cls) -> "SplitRatio":
        """Return the 3:1:1 train/valid/test ratio."""
        return cls.parse(cls.DEFAULT)

    @property
    def smallest_share(self) -> float:
        """Fraction of the data that goes to the smallest split."""
        return min(self.parts) / sum(self.parts)
