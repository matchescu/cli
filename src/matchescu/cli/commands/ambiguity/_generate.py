import re
import warnings
from collections import Counter
from collections.abc import Sequence
from functools import partial
from itertools import combinations, permutations
from os import PathLike
from typing import ClassVar

import click
from click import Context
from matchescu.matching.evaluation.data.benchmark import CsvBenchmarkData
from matchescu.reference_store.id_table import IdTable
from matchescu.typing import EntityReference
from matchescu.typing import (
    EntityReferenceIdentifier as RefId,
)
from rich.progress import Progress

from matchescu.cli.config import AmbiguityDataConfig
from matchescu.cli.runtime import get_options, make_absolute_path

from ._bridge_labels import BridgeLabels, BridgeLabelStrategy
from ._dataset_generation import DatasetGeneration
from ._pairing import GreedyPairing, Pairing, PairingStrategy
from ._ref_ids import ref_id_sort_key
from ._settings import GenerationSettings
from ._shared import (
    load_ambiguity_config,
    load_benchmark_data,
    parse_alphas,
    parse_secondary_features,
    with_profile_options,
)
from ._similarity import weighted_similarity
from ._variant import (
    RESERVED_BRIDGE_PROPERTIES,
    AmbiguousVariant,
    ComparisonData,
)
from .profiling import ProfileOptions

try:
    from rapidfuzz.distance import Levenshtein
except ImportError:
    try:
        import Levenshtein
    except ImportError:
        Levenshtein = None


TOP_N_BRIDGE_TERMS: int = 10
DEFAULT_ALPHAS: str = "0.65"


def _orient(a: RefId, b: RefId, bridges: set[RefId]) -> ComparisonData:
    return (b, a) if a in bridges and b not in bridges else (a, b)


class AmbiguityGenerator:
    """Generates ambiguous bridge records that straddle neighbouring clusters.

    For every cluster the generator nominates a representative (the member
    most ambiguous with respect to its cluster mates), pairs each
    representative with the closest other-cluster representative by ambiguity
    score, and synthesises a bridge record whose property values blend the
    two: distinct from both originals, yet claimed plausibly by either
    cluster. Cluster ground truth is derived from the input clusters, with
    every bridge joined to its home cluster and linked inside its target
    cluster.

    Values land at three distinct levels: dataset configuration supplies the
    ambiguity targets, secondary features and the minimum-ambiguity filter;
    CLI options override secondary features and supply blend parameters; the
    remaining knobs tune candidate generation (`alphas` for the harmonic-mean
    pick, `target_floor` and `target_links` for bridge linking).

    :param id_table: references of the input dataset; bridges are never
        stored here, so the id table stays bridge-free by construction.
    :param mapping_gt: true-match mapping of the input dataset; accepted to
        mirror the benchmark data interface, currently unused — the generator
        derives ground truth from `clusters` instead.
    :param clusters: ground-truth clusters of the input dataset; they receive
        ids 1..N in a deterministic sort order, and a bridge's reserved
        `target_cluster_id` property holds one of these ids.
    :param ambiguity_target_properties: properties permitted to receive
        synthesised bridged values; properties outside this list either copy
        unchanged or degrade to ``None``/empty, per type. Empty means no
        property will be bridged.
    :param secondary_features: property names copied verbatim from the home
        representative to every bridge, giving bridges authoritative
        non-target information.
    :param string_degradation_patterns: regex fragments removed before
        comparing string values; ``None`` applies the house patterns covering
        country names and business/academic suffixes.
    :param min_ambiguity: minimum pairing score for a bridge to be generated;
        ``None`` disables the filter entirely (the mode threshold analysis
        relies on).
    :param alphas: blend weights for the asymmetric harmonic-mean pick; each
        must lie in (0, 1). Every weight yields one bridge-record variant.
    :param target_floor: minimum similarity to the target reference, as a
        fraction of the symmetric pick's target similarity, in [0, 1].
    :param target_links: number of links each bridge gets inside its target
        cluster, applied in the cluster's ambiguity ranking order.
    :param pairing_strategy: selects the target cluster of each home cluster,
        for generation and for threshold analysis alike. Defaults to ``None``,
        meaning :class:`GreedyPairing`.
    :param dataset_name: name of the input dataset; hashed into string bridge
        labels so they stay reproducible and distinct across datasets.
        Defaults to ``""``, which keeps string labels unique within the
        dataset only. Integer labels do not use it.

    :raises ValueError: when any alpha lies outside (0, 1), `target_floor`
        lies outside [0, 1], `target_links` is below 1, secondary features
        overlap the ambiguity targets, or secondary features use the reserved
        reference properties (`source`, `target`, `target_cluster_id`).
    """

    _SYMMETRIC_ALPHA: ClassVar[float] = 0.5
    _DEFAULT_DEGRADATION_PATTERNS: ClassVar[list[str]] = [
        r",\s*USA",
        r",\s*US",
        r",\s*United States",
        r",\s*CA",
        r",\s*California",
        r",\s*NY",
        r",\s*New York",
        r",\s*Germany",
        r",\s*France",
        r",\s*Japan",
        r"\s+Inc\.?",
        r"\s+Corp\.?",
        r"\s+Ltd\.?",
        r"\s+University",
        r"\s+Univ\.?",
        r"\s+Research\s+Center",
        r"\s+Lab(?:oratories)?",
    ]

    def __init__(
        self,
        id_table: IdTable,
        mapping_gt: dict[ComparisonData, int],
        clusters: frozenset[frozenset[RefId]],
        ambiguity_target_properties: list[str] | None = None,
        secondary_features: list[str] | None = None,
        string_degradation_patterns: list[str] | None = None,
        min_ambiguity: float | None = None,
        alphas: tuple[float, ...] = (0.5,),
        target_floor: float = 0.0,
        target_links: int = 1,
        *,
        pairing_strategy: PairingStrategy | None = None,
        dataset_name: str = "",
    ) -> None:
        if not all(0 < alpha < 1 for alpha in alphas):
            raise ValueError("alphas must be in (0, 1)")
        if not 0 <= target_floor <= 1:
            raise ValueError("target_floor must be in [0, 1]")
        if target_links < 1:
            raise ValueError("target_links must be >= 1")
        self._mapping_gt = mapping_gt
        self._id_table = id_table
        self._cluster_count = len(clusters)
        self._cluster_id_map = {
            cluster_idx: set(cluster)
            for cluster_idx, cluster in enumerate(
                sorted(
                    clusters,
                    key=lambda c: sorted(ref_id_sort_key(r) for r in c),
                ),
                start=1,
            )
        }
        self._degradation_patterns = (
            string_degradation_patterns or self._DEFAULT_DEGRADATION_PATTERNS
        )
        self._target_properties = set(ambiguity_target_properties or [])
        self._secondary_features = set(secondary_features or [])
        self._validate_secondary_features()
        n_total_steps = 3 * self._cluster_count
        self._progress = Progress()
        self._main_task = self._progress.add_task(
            description="create ambiguous data", total=n_total_steps
        )
        self._min_ambiguity = min_ambiguity
        self._bridge_terms = self._build_bridge_terms()
        self._alphas = tuple(sorted(set(alphas)))
        self._target_floor = target_floor
        self._target_links = target_links
        self._pairing_strategy = (
            pairing_strategy if pairing_strategy is not None else GreedyPairing()
        )
        self._dataset_name = dataset_name
        self._cluster_rankings: dict[int, list[RefId]] = {}
        self.__settings = GenerationSettings(
            dataset_name=dataset_name,
            ambiguity_targets=tuple(dict.fromkeys(ambiguity_target_properties or [])),
            secondary_features=tuple(dict.fromkeys(secondary_features or [])),
            min_ambiguity=min_ambiguity,
            alphas=self._alphas,
            target_floor=target_floor,
            target_links=target_links,
        )

    @property
    def settings(self) -> GenerationSettings:
        """The parameters this generator actually uses."""
        return self.__settings

    @classmethod
    def from_config(
        cls,
        input_cfg: AmbiguityDataConfig,
        data: CsvBenchmarkData,
        *,
        secondary_features: list[str] | None,
        alphas: tuple[float, ...],
        target_floor: float,
        target_links: int,
    ) -> "AmbiguityGenerator":
        """Build a generator for one configured benchmark dataset.

        Generator settings live at two levels: the dataset configuration
        (ambiguity targets, secondary features, minimum-ambiguity filter) and
        CLI parameters (secondary features, alphas, floor, links). Both the
        `analyse` and the `generate` commands need this merge, so it lives
        here — the CLI value always wins when provided.

        :param input_cfg: one `AmbiguityConfig.input_data` entry supplying the
            ambiguity targets, dataset-level secondary features and the
            minimum-ambiguity filter.
        :param data: materialised benchmark dataset; supplies the id table,
            untouched true matches and ground-truth clusters.
        :param secondary_features: CLI-level feature names; ``None`` defers to
            the dataset configuration via the `or` fallback.
        :param alphas: harmonic-mean blend weights; the (0, 1) range is
            validated by the constructor and reported as `ValueError`.
        :param target_floor: minimum similarity to the target reference, as a
            fraction of the symmetric pick's target similarity, in [0, 1].
        :param target_links: number of links each bridge gets inside its
            target cluster; must be at least 1.
        :returns: the generator configured for exactly this dataset.
        """
        return cls(
            data.id_table,
            data.true_matches,
            data.compute_clusters(),
            input_cfg.ambiguity_targets,
            secondary_features=secondary_features or input_cfg.secondary_features,
            min_ambiguity=input_cfg.min_ambiguity,
            alphas=alphas,
            target_floor=target_floor,
            target_links=target_links,
            dataset_name=data.name,
        )

    def _validate_secondary_features(self) -> None:
        if self._secondary_features & self._target_properties:
            raise ValueError(
                "secondary features must not overlap ambiguity target properties"
            )
        reserved = self._secondary_features & RESERVED_BRIDGE_PROPERTIES
        if reserved:
            raise ValueError(
                f"secondary features must not contain reserved properties: "
                f"{sorted(reserved)}"
            )

    def _build_bridge_terms(self) -> list[str]:
        try:
            from stopwords import get_stopwords

            stop = {w.lower() for w in get_stopwords("en")}
        except Exception:  # noqa: BLE001
            stop = set()

        token_re = re.compile(r"\b\w{2,}\b")

        # tf[doc_id][term] = count; df[term] = number of docs containing term
        tf: dict[str, dict[str, int]] = {}
        df: dict[str, int] = {}

        for ref in self._id_table:
            doc_id = f"{ref.id.label}-{ref.id.source}"
            doc_tf: dict[str, int] = {}
            for val in ref.as_dict().values():
                if not isinstance(val, str):
                    continue
                for tok in token_re.findall(val):
                    tok_l = tok.lower()
                    if tok_l in stop:
                        continue
                    doc_tf[tok_l] = doc_tf.get(tok_l, 0) + 1
            if not doc_tf:
                continue
            tf[doc_id] = doc_tf
            for term in doc_tf:
                df[term] = df.get(term, 0) + 1

        if not tf or not df:
            return []

        # Score = (total TF across corpus) / DF  — rewards common-but-not-ubiquitous terms
        scores: dict[str, float] = {}
        for doc_tf in tf.values():
            for term, count in doc_tf.items():
                scores[term] = scores.get(term, 0.0) + count / df[term]

        ranked = sorted(scores, key=lambda t: scores[t], reverse=True)
        return [t for t in ranked[:TOP_N_BRIDGE_TERMS]]

    def degrade_str(self, input_str) -> str:
        """Remove all degradation patterns from a property value.

        Degraded references keep enough of both originals to be claimed by
        either cluster; removing the configured suffixes early makes values
        comparable across dataset vocabularies. Removal artefacts (repeated
        commas, stray edge commas, collapsed spaces) are normalised
        afterwards.

        :param input_str: property value to degrade.
        :returns: the value with every pattern removed case-insensitively and
            commas/whitespace normalised.
        """
        result = input_str
        for pattern in self._degradation_patterns:
            result = re.sub(pattern, "", result, flags=re.IGNORECASE)

        # Clean up extra spaces and commas
        result = re.sub(r"\s*,\s*,", ", ", result)
        result = re.sub(r"^\s*,\s*|\s*,\s*$", "", result)
        result = re.sub(r"\s+", " ", result).strip()

        return result

    def _select_cluster_representative(self, cluster, selector) -> str:
        prev_degraded = ""
        for ref in cluster:
            initial_value = selector(ref)
            degraded = self.degrade_str(initial_value)
            if len(degraded) <= 5:
                continue
            if degraded.lower() == initial_value.lower():
                continue
            if len(degraded) < len(prev_degraded) or len(prev_degraded) == 0:
                prev_degraded = degraded
        return prev_degraded

    @classmethod
    def _edit_distance(cls, s1: str, s2: str) -> int:
        if Levenshtein is not None:
            return Levenshtein.distance(s1, s2)
        if len(s1) < len(s2):
            return cls._edit_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Cost is 0 if characters match, 1 otherwise
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (0 if c1 == c2 else 1)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    @classmethod
    def _edit_similarity(cls, s1: str, s2: str) -> float:
        if s1 == s2:
            return 1.0

        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0 if s1 == s2 else 0.0

        distance = cls._edit_distance(s1, s2)

        return 1.0 - (distance / max_len)

    def _ambiguity_score(self, x: EntityReference, y: EntityReference) -> float:
        # Identical values are ideal bridging material (+2.0); differing
        # strings on target properties add at most 2x inverted edit
        # similarity, while differences on non-target properties lose
        # information (-0.5), as do values present on one side only (which
        # the bridge must remove as differentiators) and type mismatches or
        # non-string differences (-1.0) — material the matching model would
        # discriminate easily.
        score = 0.0

        x_dict = x.as_dict()
        y_dict = y.as_dict()
        all_keys = set(x_dict.keys()) | set(y_dict.keys())

        for key in all_keys:
            if (key in x_dict) ^ (key in y_dict):
                # the ambiguous reference will need to remove differentiators
                score -= 0.5
                continue

            val_a = x_dict[key]
            val_b = y_dict[key]

            if val_a == val_b:
                # all identical values are wonderful for creating ambiguity
                score += 2.0
            elif isinstance(val_a, str) and isinstance(val_b, str):
                val_a = val_a.lower().strip()
                val_b = val_b.lower().strip()
                if key in self._target_properties:
                    # obtaining an ambiguous entity reference is still desirable
                    # when the key is part of the targeted properties
                    increment = 0
                    if val_a in val_b or val_b in val_a:
                        increment = 2.0
                    else:
                        increment = 2.0 * self._edit_similarity(val_a, val_b)
                    score += increment
                else:
                    # non-target different strings become 'None' (information loss)
                    score -= 0.5
            else:
                # Type mismatch or non-string difference results in 'None', but the
                # matching model would probably have an easier time discriminating
                # if there's a type mismatch between the bridged entity references
                score -= 1.0

        return score

    def _get_cluster_representatives(self):
        cluster_reps = {}
        self._cluster_rankings = {}
        for cluster_id, ref_ids in self._cluster_id_map.items():
            self._progress.update(
                self._main_task, description=f"nominating [{cluster_id}] cluster rep"
            )
            candidates = sorted(ref_ids, key=ref_id_sort_key)
            refs = list(self._id_table.get_all(candidates))
            rep_idx = 0
            mean_ambiguities = None
            if len(refs) > 1:
                mean_ambiguities = []
                for i, ref in enumerate(refs):
                    others = refs[:i] + refs[i + 1 :]
                    n = len(others)
                    ambiguity_scores = [self._ambiguity_score(ref, x) for x in others]
                    mean_score = sum(ambiguity_scores) / n if n > 0 else 0
                    mean_ambiguities.append(mean_score)
                max_ambiguity = 0
                rep_idx = 0
                for i, ambiguity in enumerate(mean_ambiguities):
                    if max_ambiguity < ambiguity:
                        max_ambiguity = ambiguity
                        rep_idx = i
            cluster_reps[cluster_id] = candidates[rep_idx]
            self._cluster_rankings[cluster_id] = self._ranked_by_amb(
                candidates, rep_idx, mean_ambiguities
            )
            self._progress.advance(self._main_task)
        return cluster_reps

    @staticmethod
    def _ranked_by_amb(clust_members, rep_idx, mean_ambiguities):
        ranking = [clust_members[rep_idx]]
        if mean_ambiguities is None:
            return ranking  # no mean ambiguities -> cluster rep only
        others = sorted(
            (
                (mean_ambiguities[i], clust_members[i])
                for i in range(len(clust_members))
                if i != rep_idx
            ),
            key=lambda tpl: -tpl[0],
        )
        ranking.extend(ref_id for _, ref_id in others)
        return ranking

    def _rep_refs(self, cluster_reps) -> dict[int, EntityReference]:
        return {
            cluster_id: self._id_table.get(ref_id)
            for cluster_id, ref_id in cluster_reps.items()
        }

    def _rep_affinity(self, rep_refs, home: int, other: int) -> float:
        return self._ambiguity_score(rep_refs[home], rep_refs[other])

    def _on_pairing_home(self, cluster_id: int) -> None:
        self._progress.update(
            self._main_task,
            description=f"finding [{cluster_id}] closest cluster rep",
        )
        self._progress.advance(self._main_task)

    def _pairings(self, rep_refs, min_score: float | None) -> list[Pairing]:
        return self._pairing_strategy(
            list(rep_refs),
            partial(self._rep_affinity, rep_refs),
            min_score,
            self._on_pairing_home,
        )

    def _find_closest_rep(self, cluster_reps):
        rep_refs = self._rep_refs(cluster_reps)
        pairings = self._pairings(rep_refs, self._min_ambiguity)
        return {
            pairing.home_cluster_id: (
                pairing.target_cluster_id,
                pairing.score,
                rep_refs[pairing.target_cluster_id],
            )
            for pairing in pairings
        }

    @classmethod
    def _score_pool(cls, val_a, val_b, bridge_terms) -> list[tuple[str, float, float]]:
        # (candidate, home, target) sorted by candidate; candidates equal to
        # an input (case-insensitive) score (0.0, 0.0).
        words = list(
            dict.fromkeys(w for w in val_a.split() + val_b.split() if len(w) > 2)
        )
        bt = list(dict.fromkeys(list(bridge_terms or []) + words))
        lo = (val_a.lower(), val_b.lower())
        out = []
        for c in cls._candidates(val_a, val_b, bt):
            if c.lower() in lo:
                out.append((c, 0.0, 0.0))
            else:
                out.append(
                    (
                        c,
                        weighted_similarity(c, val_a),
                        weighted_similarity(c, val_b),
                    )
                )
        return sorted(out)

    @staticmethod
    def _pick(scored, alpha: float, floor: float = 0.0):
        # Weighted harmonic mean of home and target similarity:
        # 2*s_h*s_t / (s_t*alpha + s_h*(1-alpha)), alpha in (0, 1);
        # alpha = 0.5 degenerates to the symmetric mean. The target floor
        # keeps candidates that still resemble the target cluster.
        # Returns (candidate, harmonic_mean, sim_home, sim_target); None when
        # nothing clears the floor.
        best, best_h = None, -1.0
        for c, sa, sb in scored:
            if sb < floor:
                continue
            h = 0.0
            if sa > 0 and sb > 0:
                denominator = (alpha / sa) + (1 - alpha) / sb
                h = 1.0 / denominator
            if h > best_h:
                best, best_h = (c, h, sa, sb), h
        return best

    @staticmethod
    def _fallback_text(val_a: str, val_b: str) -> str:
        uq_words = dict.fromkeys(w for w in val_a.split() + val_b.split() if len(w) > 2)
        for word in uq_words:
            if word.lower() not in (val_a.lower(), val_b.lower()):
                return word
        return val_a

    @classmethod
    def _cpfx(cls, a: str, b: str) -> str:
        i = 0
        for x, y in zip(a.lower(), b.lower()):
            if x != y:
                break
            i += 1
        return a[:i]

    @classmethod
    def _blend(cls, a: str, b: str) -> str:
        return "".join(
            (
                (a[i] if i % 2 == 0 else b[i])
                if i < len(a) and i < len(b)
                else (a[i] if i < len(a) else b[i])
            )
            for i in range(max(len(a), len(b)))
        )

    @classmethod
    def _candidates(cls, a: str, b: str, bridge: list[str]) -> set[str]:
        wa, wb = a.split(), b.split()
        la, lb = {w.lower() for w in wa}, {w.lower() for w in wb}
        cs = la & lb

        common = [w for w in wa if w.lower() in cs]
        ua = [w for w in wa if w.lower() not in cs]
        ub = [w for w in wb if w.lower() not in cs]
        base = " ".join(common)

        raw_pool: list[str] = []

        # 1. Base + every prefix-truncation of each unique word
        if base:
            raw_pool.append(base)
        for w in ua + ub:
            for k in range(1, len(w)):
                raw_pool.append(f"{base} {w[:k]}")

        # 2. Pair unique words -> common-prefix extensions / blend / mid-cut
        for wa_ in ua:
            for wb_ in ub:
                cp = cls._cpfx(wa_, wb_)
                if cp:
                    raw_pool.append(f"{base} {cp}")
                    for e in range(1, max(len(wa_), len(wb_)) - len(cp) + 1):
                        for w in (wa_, wb_):
                            if len(cp) + e <= len(w):
                                raw_pool.append(f"{base} {w[: len(cp) + e]}")
                raw_pool.append(f"{base} {cls._blend(wa_, wb_)}")
                mid = (len(wa_) + len(wb_)) // 2
                if mid:
                    raw_pool.append(f"{base} {wa_[:mid]}")
                    raw_pool.append(f"{base} {wb_[:mid]}")

        # 3. Cross-structural mixing (comma-separated hybrids)
        if ua and ub:
            raw_pool.append(f"{', '.join(ua)}, {b}")
            raw_pool.append(f"{', '.join(ub)}, {a}")
            for x in ua[:2]:
                for y in ub[:2]:
                    raw_pool.append(f"{x}, {' '.join(common + [y])}")
                    raw_pool.append(f"{y}, {' '.join(common + [x])}")

        # 4. Word substitution inside each input's frame
        for i, w in enumerate(wa):
            for ww in wb:
                if w.lower() != ww.lower():
                    raw_pool.append(" ".join(wa[:i] + [ww] + wa[i + 1 :]))
        for i, w in enumerate(wb):
            for ww in wa:
                if w.lower() != ww.lower():
                    raw_pool.append(" ".join(wb[:i] + [ww] + wb[i + 1 :]))

        # 5. Subsets & (small) permutations of the word union
        union = list(dict.fromkeys(wa + wb))
        cap = min(len(union) + 1, 5)
        for r in range(1, cap):
            for c in combinations(union, r):
                raw_pool.append(" ".join(c))
            if len(union) <= 6:
                for p in permutations(union, r):
                    raw_pool.append(" ".join(p))

        # 6. Domain bridge terms
        for t in bridge:
            if base:
                raw_pool.append(f"{base} {t}")
            for w in (ua + ub)[:2]:
                raw_pool.append(f"{w} {t}")

        # 7. Containment: gradual extension of the shorter string
        lo_a, lo_b = a.lower(), b.lower()
        if lo_a.startswith(lo_b) or lo_b.startswith(lo_a):
            longer, shorter = (a, b) if len(a) >= len(b) else (b, a)
            extra = longer[len(shorter) :].strip()
            for k in range(1, len(extra) + 1):
                raw_pool.append(f"{shorter} {extra[:k]}")

        # Whitespace-normalise each candidate, dropping empties and anything
        # identical to the original values.
        return {
            normalized
            for c in raw_pool
            if (normalized := " ".join(c.split())) and normalized not in (a, b)
        }

    @staticmethod
    def _copy_secondary_feature(
        source_dict: dict,
        target_dict: dict,
        key: str,
        counts: Counter,
    ):
        home_value = source_dict.get(key, None) or None
        if home_value is None:
            counts["missing"] += 1
            warnings.warn(
                f"secondary feature '{key}' missing or empty on home representative; skipping"
            )
            return None
        counts["copied"] += 1
        if key in target_dict and target_dict[key] == home_value:
            counts["equal"] += 1
        return home_value

    def _print_secondary_feature_summary(self, counts: dict[str, Counter]) -> None:
        if not self._secondary_features:
            return
        for key in sorted(counts):
            stats = counts[key]
            print(
                f"secondary feature [{key}]: copied={stats['copied']}, "
                f"missing={stats['missing']}, equal={stats['equal']}"
            )

    def _generate_ambiguous_references(
        self, cluster_reps, closest_correspondents, next_label: BridgeLabelStrategy
    ):
        start_refs = {k: self._id_table.get(v) for k, v in cluster_reps.items()}
        secondary_counts = {k: Counter() for k in self._secondary_features}

        ambiguous_refs = {alpha: {} for alpha in self._alphas}
        bridge_logs: dict[float, list[dict]] = {alpha: [] for alpha in self._alphas}
        for cluster_id, start_ref in start_refs.items():
            self._progress.update(
                self._main_task,
                description=f"generating ambiguous cluster [{cluster_id}] reference",
            )
            if cluster_id not in closest_correspondents:
                self._progress.advance(self._main_task)
                continue
            target_cluster_id, _score, target = closest_correspondents[cluster_id]
            bridge_label = next_label(cluster_id)
            source_dict = start_ref.as_dict()
            target_dict = target.as_dict()
            all_keys = set(source_dict.keys()) | set(target_dict.keys())
            per_alpha_props = {
                alpha: {
                    "source": start_ref,
                    "target": target,
                    "target_cluster_id": target_cluster_id,
                }
                for alpha in self._alphas
            }
            for k in sorted(all_keys):
                if k in self._secondary_features:
                    home_value = self._copy_secondary_feature(
                        source_dict, target_dict, k, secondary_counts[k]
                    )
                    if home_value is not None:
                        for props in per_alpha_props.values():
                            props[k] = home_value
                    continue
                # attribute in one but not the other -> continue
                if (k in source_dict) ^ (k in target_dict):
                    continue
                val_a = start_ref[k]
                val_b = target[k]
                if val_a == val_b:
                    for props in per_alpha_props.values():
                        props[k] = val_a
                elif isinstance(val_a, str) and isinstance(val_b, str):
                    if k not in self._target_properties:
                        for props in per_alpha_props.values():
                            props[k] = ""
                    else:
                        origin = {
                            "bridge_label": bridge_label,
                            "home_cluster": cluster_id,
                            "target_cluster": target_cluster_id,
                            "field": k,
                        }
                        for alpha, entry in self._bridged_values(
                            val_a, val_b, origin
                        ).items():
                            per_alpha_props[alpha][k] = entry["value"]
                            bridge_logs[alpha].append(entry)
                else:
                    for props in per_alpha_props.values():
                        props[k] = None
            for alpha, props in per_alpha_props.items():
                new_ref = EntityReference(
                    RefId(bridge_label, "ambiguity-generator"), props
                )
                ambiguous_refs[alpha][cluster_id] = new_ref
            self._progress.advance(self._main_task)
        self._print_secondary_feature_summary(secondary_counts)
        return ambiguous_refs, bridge_logs

    def _bridged_values(self, val_a: str, val_b: str, origin: dict) -> dict:
        scored = self._score_pool(val_a, val_b, self._bridge_terms)
        sym = self._pick(scored, self._SYMMETRIC_ALPHA)
        floor = self._target_floor * sym[3] if sym else 0.0
        return {
            alpha: self._bridge_log_entry(
                alpha,
                origin,
                self._pick(scored, alpha, floor),
                self._pick(scored, alpha),
                (val_a, val_b),
            )
            for alpha in self._alphas
        }

    def _bridge_log_entry(self, alpha, origin, pick, unfloored, values) -> dict:
        sim_home, sim_target = (pick[2], pick[3]) if pick else (None, None)
        return {
            "alpha": alpha,
            **origin,
            "value": self._picked_value(pick, values),
            "value_no_floor": self._picked_value(unfloored, values),
            "sim_home": sim_home,
            "sim_target": sim_target,
            "share_home": self._share_home(pick),
            "fallback": pick is None,
        }

    def _picked_value(self, pick, values: tuple[str, str]) -> str:
        return pick[0] if pick else self._fallback_text(*values)

    @staticmethod
    def _share_home(pick) -> float | None:
        if pick is None or pick[2] + pick[3] <= 0:
            return None
        return pick[2] / (pick[2] + pick[3])

    def _build_ground_truth(self, bridges, alpha):
        cluster_gt: dict[int, set[RefId]] = {
            cid: set(refs) for cid, refs in self._cluster_id_map.items()
        }
        for cluster_id, ref in bridges.items():
            cluster_gt[cluster_id].add(ref.id)

        directed_gt: dict[ComparisonData, int] = {}
        undirected_gt: dict[ComparisonData, int] = {}
        link_log: list[dict] = []
        bridge_ids = {ref.id for ref in bridges.values()}
        for cluster_id, ref_ids in cluster_gt.items():
            sorted_ref_ids = sorted(ref_ids, key=ref_id_sort_key)
            for left, right in combinations(sorted_ref_ids, 2):
                cmp_data = _orient(left, right, bridge_ids)
                undirected_gt[cmp_data] = 1
                directed_gt[cmp_data] = 1
            if (bridge := bridges.get(cluster_id)) is not None:
                for rank, t_id in enumerate(
                    self._cluster_rankings[bridge.target_cluster_id][
                        : self._target_links
                    ]
                ):
                    cmp_data = (t_id, bridge.id)
                    undirected_gt[cmp_data] = 1
                    directed_gt[cmp_data] = 2
                    link_log.append(
                        {
                            "alpha": alpha,
                            "bridge_label": bridge.id.label,
                            "target_cluster": bridge.target_cluster_id,
                            "target_label": t_id.label,
                            "rank": rank,
                            "ambiguity": self._ambiguity_score(
                                bridge, self._id_table.get(t_id)
                            ),
                        }
                    )
        return cluster_gt, directed_gt, undirected_gt, link_log

    def _variant(self, alpha: float, bridges, bridge_log) -> AmbiguousVariant:
        cluster_gt, directed_gt, undirected_gt, link_log = self._build_ground_truth(
            bridges, alpha
        )
        return AmbiguousVariant(
            alpha=alpha,
            bridges=tuple(bridges.values()),
            cluster_gt=cluster_gt,
            directed_gt=directed_gt,
            undirected_gt=undirected_gt,
            bridge_log=tuple(bridge_log),
            link_log=tuple(link_log),
        )

    def best_pairing_scores(self) -> list[float]:
        """Return the score of every pairing made without the threshold filter.

        Threshold analysis needs a score population that does not depend on
        the configured operating point, so `min_ambiguity` is ignored here.

        .. note::
            The scores come from the configured pairing strategy. With the
            default :class:`GreedyPairing`, homes are processed in cluster-id
            order and a cluster accepted as a home can no longer be a target,
            so a later home's best partner may already be taken and the last
            home may get no pairing at all. The scores are therefore the
            scores of the pairing the strategy makes, not each cluster's
            globally best score.

        :returns: one score per paired home cluster, in cluster-id order.
        """
        rep_refs = self._rep_refs(self._get_cluster_representatives())
        return [pairing.score for pairing in self._pairings(rep_refs, None)]

    def bridge_counts(self, min_scores: Sequence[float | None]) -> list[int]:
        """Count how many bridges would be produced at each threshold.

        Threshold analysis needs the number of bridges a run with
        ``min_ambiguity = t`` would actually generate. Counting pairing scores
        at or above ``t`` is not enough: a home rejected by the threshold
        stays available as a target, which changes the pairings of later
        homes. Each threshold therefore gets its own pairing run, and the
        cost grows linearly with the number of thresholds.

        .. note::
            Counts are not necessarily monotonic in the threshold: raising it
            can free a target that a later home then pairs with.

        :param min_scores: thresholds to evaluate, each standing in for
            `min_ambiguity`; ``None`` means no threshold.
        :returns: one bridge count per threshold, in the same order.
        """
        rep_refs = self._rep_refs(self._get_cluster_representatives())
        return [len(self._pairings(rep_refs, t)) for t in min_scores]

    def __call__(self) -> dict[float, AmbiguousVariant]:
        """Generate every alpha variant of the ambiguous bridge records.

        Streams the three stages (representative nomination, closest-cluster
        pairing, bridge synthesis) through the shared rich progress bar, then
        derives the per-alpha ground truth from clusters and the bridges.
        Every call starts from scratch, so repeated calls return equal
        variants, and a variant does not depend on the other alphas
        generated alongside it.

        :returns: one :class:`AmbiguousVariant` per alpha weight. Bridges are
            intentionally absent from the id table. Bridge labels come from
            :class:`BridgeLabels`: integers continue after the largest
            existing label, strings are reproducible UUID5 values.
        :raises TypeError: when the dataset's labels mix types or have a type
            no bridge label strategy supports; raised before any work starts.
        """
        next_label = BridgeLabels.labelling_strategy(
            [ref.id.label for ref in self._id_table], self._dataset_name
        )
        self._progress.start()
        try:
            cluster_reps = self._get_cluster_representatives()
            closest_reps = self._find_closest_rep(cluster_reps)
            ambiguous_refs, bridge_logs = self._generate_ambiguous_references(
                cluster_reps, closest_reps, next_label
            )
            return {
                alpha: self._variant(alpha, ambiguous_refs[alpha], bridge_logs[alpha])
                for alpha in self._alphas
            }
        finally:
            self._progress.stop()


@click.command("generate")
@click.option(
    "-c",
    "--config-file",
    required=True,
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True
    ),
    help="configuration file for the ambiguity generator",
)
@click.option(
    "-O",
    "--output-dir",
    required=True,
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, writable=True, resolve_path=True
    ),
    help="root directory where ambiguous datasets corresponding to the configured input datasets are generated",
)
@click.option(
    "--secondary-features",
    "secondary_features",
    default=None,
    help="Comma-separated secondary features to copy from the home representative.",
)
@click.option(
    "--alphas",
    "alphas",
    type=click.STRING,
    default=DEFAULT_ALPHAS,
    required=False,
    help="Comma-separated alpha weights (in (0, 1)) for the harmonic-mean pick; one dataset variant per weight",
)
@click.option(
    "--target-floor",
    "target_floor",
    type=click.FLOAT,
    default=0.0,
    required=False,
    help="Floor on sim_target, as a fraction of the symmetric-pick sim_target (in [0, 1])",
)
@click.option(
    "--target-links",
    "target_links",
    type=click.INT,
    default=1,
    required=False,
    help="How many links each bridge gets inside its target cluster",
)
@with_profile_options
@click.pass_context
def generate_ambiguous_datasets(
    ctx: Context,
    config_file: str | PathLike,
    output_dir: str | PathLike,
    secondary_features: str | None,
    alphas: str,
    target_floor: float,
    target_links: int,
    profile_options: ProfileOptions,
):
    """Introduce ambiguous data into existing dataset.

    Besides the variants, every run upserts one row per variant into
    ``metadata.csv`` and writes the bridge components table, both in the
    output directory.
    """
    root_dir = get_options(ctx).root_dir
    config = load_ambiguity_config(config_file, root_dir)
    output_dir = make_absolute_path(output_dir, root_dir)
    cli_secondary_features = parse_secondary_features(secondary_features)
    alphas_tuple = parse_alphas(alphas)
    generation = DatasetGeneration(output_dir, profile_options)

    for input_cfg in config.input_data:
        print("processing", input_cfg.dataset.directory)
        data = load_benchmark_data(input_cfg, root_dir)
        generator = AmbiguityGenerator.from_config(
            input_cfg,
            data,
            secondary_features=cli_secondary_features,
            alphas=alphas_tuple,
            target_floor=target_floor,
            target_links=target_links,
        )
        generation.run(generator, data.id_table)
