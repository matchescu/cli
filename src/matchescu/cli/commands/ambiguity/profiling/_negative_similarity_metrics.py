from random import Random
from typing import Any

from matchescu.typing import EntityReferenceIdentifier as RefId

from ._context import ProfileContext
from ._input import RecordPair
from ._stats import auc, spread


class NegativeSimilarityMetrics:
    """Content similarity of implicit label-0 pairs, against the positives.

    Label-0 pairs are never listed, so a seeded sample is drawn uniformly
    from the unordered record pairs that join two different clusters and
    are not positive in either orientation. The sample holds
    ``neg_pos_ratio`` label-0 pairs per positive pair, rounded; drawing
    stops early once every pair has been drawn, so a ratio beyond the
    label-0 population yields the whole population.
    """

    def __call__(self, context: ProfileContext) -> dict[str, Any]:
        negatives = self.__sample(context)
        score = context.profile_input.score
        label0 = [score(left, right) for left, right in negatives]
        roles = context.roles
        label1 = [score(left, right) for left, right in (*roles.source, *roles.extra)]
        label2 = [score(left, right) for left, right in roles.target]
        return {
            "neg_sample_size": len(negatives),
            **spread("sim_label0", label0),
            "content_auc_label1_vs_label0": auc(label1, label0),
            "content_auc_label2_vs_label0": auc(label2, label0),
        }

    def __sample(self, context: ProfileContext) -> list[RecordPair]:
        refs = sorted(context.profile_input.records, key=self.__sort_key)
        n_pairs = len(refs) * (len(refs) - 1) // 2
        rng = Random(context.options.seed)
        drawn: set[tuple[int, int]] = set()
        negatives: list[RecordPair] = []
        size = round(
            context.options.neg_pos_ratio * len(context.profile_input.directed_gt)
        )
        while len(negatives) < size and len(drawn) < n_pairs:
            i, j = sorted(rng.sample(range(len(refs)), 2))
            if (i, j) in drawn:
                continue
            drawn.add((i, j))
            if self.__is_negative(context, refs[i], refs[j]):
                negatives.append((refs[i], refs[j]))
        return negatives

    @staticmethod
    def __is_negative(context: ProfileContext, left: RefId, right: RefId) -> bool:
        profile_input = context.profile_input
        positives = profile_input.directed_gt
        return (
            profile_input.cluster_of[left] != profile_input.cluster_of[right]
            and (left, right) not in positives
            and (right, left) not in positives
        )

    @staticmethod
    def __sort_key(ref_id: RefId) -> tuple[str, str]:
        return (ref_id.source, str(ref_id.label))
