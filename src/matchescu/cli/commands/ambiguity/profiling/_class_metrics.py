from collections import Counter
from math import comb
from typing import Any

from ._context import ProfileContext
from ._pair_roles import PairRoles
from ._stats import ratio

PAIR_SIZE: int = 2


class ClassMetrics:
    """Class counts and imbalance over the full comparison space.

    Label 0 is implicit: every unordered record pair that is not a positive
    pair. The names mirror the split configuration (`neg_pos_imbalance` for
    ``negPosRatio``, `match_bridge_ratio` for ``matchBridgeRatio``).
    """

    def __call__(self, context: ProfileContext) -> dict[str, Any]:
        labels = Counter(context.profile_input.directed_gt.values())
        n_label1 = labels[PairRoles.MATCH]
        n_label2 = labels[PairRoles.DIRECTIONAL_MATCH]
        n_positive = n_label1 + n_label2
        space = comb(len(context.profile_input.records), PAIR_SIZE)
        n_negative = space - n_positive
        return {
            "n_label1": n_label1,
            "n_label2": n_label2,
            "n_undirected": n_positive,
            "comparison_space_size": space,
            "n_label0_implicit": n_negative,
            "positive_rate": ratio(n_positive, space),
            "neg_pos_imbalance": ratio(n_negative, n_positive),
            "match_bridge_ratio": ratio(n_label1, n_label2),
        }
