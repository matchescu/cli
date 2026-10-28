from dataclasses import dataclass, field

from ._split_ratio import SplitRatio


@dataclass(frozen=True)
class ProfileOptions:
    """Settings that change what a profile computes.

    :ivar split_ratio: split sizes the bridge component split load is
        measured against.
    """

    split_ratio: SplitRatio = field(default_factory=SplitRatio.default)
