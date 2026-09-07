import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import combinations, permutations
from os import PathLike
from pathlib import Path
from typing import Any, ClassVar

import click
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from click import Context
from matchescu.matching.evaluation.data.benchmark import CsvBenchmarkData
from matchescu.reference_store.id_table import IdTable
from matchescu.typing import (
    EntityReference,
)
from matchescu.typing import (
    EntityReferenceIdentifier as RefId,
)
from polars import DataFrame
from rich.progress import Progress
from scipy.optimize import brentq
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

from matchescu.cli._cmd_group import matchescu
from matchescu.cli.config import AmbiguityConfig, JSONConfig, new_benchmark_data_factory
from matchescu.cli.runtime import get_options, make_absolute_path

try:
    from rapidfuzz.distance import Levenshtein
except ImportError:
    try:
        import Levenshtein
    except ImportError:
        Levenshtein = None


DIR = Path("./data/affiliationstrings/")
TOP_N_BRIDGE_TERMS: int = 10

type ComparisonData = tuple[RefId, RefId]


def _ref_id_sort_key(ref_id: RefId):
    """Stable, process-independent sort key for ``EntityReferenceIdentifier``."""
    return (ref_id.source, ref_id.label)


@dataclass
class GmmFit:
    """Parameters of the two-component GMM fit used to derive the ambiguity threshold.

    ``threshold`` is the exact (unrounded) score at which the posterior probability
    of the genuine-overlap component equals 0.5 (the Bayes-optimal decision boundary),
    or the midpoint of the two component means when the components are not separable.
    The ``boundary_type`` field distinguishes the two cases so that degenerate fits
    can be reported honestly rather than silently.
    """

    threshold: float
    mu_genuine: float
    sigma_genuine: float
    pi_genuine: float
    mu_superficial: float
    sigma_superficial: float
    pi_superficial: float
    boundary_type: str  # "bayes_optimal" | "mean_midpoint"


class AmbiguityGenerator:
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
        string_degradation_patterns: list[str] | None = None,
        min_ambiguity: float | None = None,
    ) -> None:
        self._mapping_gt = mapping_gt
        self._id_table = id_table
        self._cluster_count = len(clusters)
        self._cluster_id_map = {
            cluster_idx: set(cluster)
            for cluster_idx, cluster in enumerate(
                sorted(
                    clusters,
                    key=lambda c: sorted(_ref_id_sort_key(r) for r in c),
                ),
                start=1,
            )
        }
        self._degradation_patterns = (
            string_degradation_patterns or self._DEFAULT_DEGRADATION_PATTERNS
        )
        self._target_properties = set(ambiguity_target_properties or [])
        n_total_steps = 3 * self._cluster_count
        self._progress = Progress()
        self._main_task = self._progress.add_task(
            description="create ambiguous data", total=n_total_steps
        )
        self._min_ambiguity = min_ambiguity
        self._bridge_terms = self._build_bridge_terms()

    def _build_bridge_terms(self) -> list[str]:
        """Build dataset-specific bridge terms using TF/IDF on the id table.

        Steps:
        1. Tokenise every string field value; skip stopwords.
        2. Count term frequency (TF) per document and document frequency (DF).
        3. Rank by TF-IDF ratio; return the top TOP_N_BRIDGE_TERMS terms.
        """
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

        len(tf)
        # Score = (total TF across corpus) / DF  — rewards common-but-not-ubiquitous terms
        scores: dict[str, float] = {}
        for doc_tf in tf.values():
            for term, count in doc_tf.items():
                scores[term] = scores.get(term, 0.0) + count / df[term]

        ranked = sorted(scores, key=lambda t: scores[t], reverse=True)
        return [t for t in ranked[:TOP_N_BRIDGE_TERMS]]

    def degrade_str(self, input_str):
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
        """Compute how ambiguous the result of combining ``x`` and ``y`` would be.

        The algorithm for computing ambiguity uses all of the property names provided
        by the user to compare x and y. For each property value it applies the
        following rules:

        - identical property values get +2.0
        - 2.0 * inv(edit distance) if property values are not identical
        - property is not a string gets -0.5
        - type mismatch gets -1.0
        - property does not exist in ``x`` or ``y`` other gets -0.5

        :param x: first entity reference
        :param y: second entity reference
        :params property_names: variable array of property names to consider when
        computing ambiguity; these are the properties that would likely yield the
        ambiguous values.

        :return: an ambiguity score: the higher, the more chances of being able to
        generate an ambiguous entity reference.
        """
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
        for cluster_id, ref_ids in self._cluster_id_map.items():
            self._progress.update(
                self._main_task, description=f"nominating [{cluster_id}] cluster rep"
            )
            candidates = sorted(ref_ids, key=_ref_id_sort_key)
            refs = list(self._id_table.get_all(candidates))
            rep_idx = 0
            if len(refs) > 1:
                mean_ambiguities = []
                for i, ref in enumerate(refs):
                    others = refs[:i] + refs[i + 1 :]
                    ambiguity_scores = [self._ambiguity_score(ref, x) for x in others]
                    mean_ambiguities.append(
                        sum(ambiguity_scores) / len(ambiguity_scores)
                    )
                max_ambiguity = 0
                rep_idx = 0
                for i, ambiguity in enumerate(mean_ambiguities):
                    if max_ambiguity < ambiguity:
                        max_ambiguity = ambiguity
                        rep_idx = i
            cluster_reps[cluster_id] = candidates[rep_idx]
            self._progress.advance(self._main_task)
        return cluster_reps

    def _find_closest_rep(self, cluster_reps):
        cluster_refs = {
            cluster_id: self._id_table.get(ref_id)
            for cluster_id, ref_id in cluster_reps.items()
        }
        result = {}
        for cluster_id, ref in cluster_refs.items():
            self._progress.update(
                self._main_task,
                description=f"finding [{cluster_id}] closest cluster rep",
            )
            if cluster_id in result:
                self._progress.advance(self._main_task)
                continue
            other_refs = {
                other_id: other_ref
                for other_id, other_ref in cluster_refs.items()
                if other_id != cluster_id and other_id not in result
            }
            if len(other_refs) < 1:
                break

            best_ambiguity = None
            best_id = None
            for other_id, other_ref in other_refs.items():
                score = self._ambiguity_score(ref, other_ref)
                if best_ambiguity is None or score > best_ambiguity:
                    best_ambiguity = score
                    best_id = other_id
                elif (
                    score == best_ambiguity
                    and _ref_id_sort_key(other_id) < _ref_id_sort_key(best_id)
                ):
                    best_id = other_id
            if self._min_ambiguity is None or best_ambiguity >= self._min_ambiguity:
                result[cluster_id] = (best_id, best_ambiguity, other_refs[best_id])
            self._progress.advance(self._main_task)

        return result

    @classmethod
    def _ngram_sim(cls, a: str, b: str, n: int = 3) -> float:
        """Character n-gram Jaccard similarity."""
        a, b = a.lower(), b.lower()
        if len(a) < n or len(b) < n:
            sa, sb = set(a.replace(" ", "")), set(b.replace(" ", ""))
            return len(sa & sb) / max(len(sa | sb), 1)
        na = {a[i : i + n] for i in range(len(a) - n + 1)}
        nb = {b[i : i + n] for i in range(len(b) - n + 1)}
        return len(na & nb) / max(len(na | nb), 1)

    @classmethod
    def _tok_sim(cls, a: str, b: str) -> float:
        """Word-level Jaccard similarity."""
        wa, wb = set(a.lower().split()), set(b.lower().split())
        return len(wa & wb) / max(len(wa | wb), 1)

    @classmethod
    def _weighted_similarity(cls, a: str, b: str) -> float:
        """Weighted composite: 40 % char-ngram, 30 % token, 30 % sequence."""
        return (
            0.4 * cls._ngram_sim(a, b)
            + 0.3 * cls._tok_sim(a, b)
            + 0.3 * SequenceMatcher(None, a.lower(), b.lower()).ratio()
        )

    @classmethod
    def _score(cls, cand: str, a: str, b: str) -> tuple[float, float, float]:
        """(harmonic_mean, sim_a, sim_b) — 0 if cand equals either input."""
        if cand.lower() in (a.lower(), b.lower()):
            return 0.0, 0.0, 0.0
        sa, sb = cls._weighted_similarity(cand, a), cls._weighted_similarity(cand, b)
        h = 2 * sa * sb / (sa + sb) if sa + sb else 0.0
        return h, sa, sb

    @classmethod
    def _cpfx(cls, a: str, b: str) -> str:
        """Longest common case-insensitive prefix, preserving case from *a*."""
        i = 0
        for x, y in zip(a.lower(), b.lower()):
            if x != y:
                break
            i += 1
        return a[:i]

    @classmethod
    def _blend(cls, a: str, b: str) -> str:
        """Alternate characters: even indices from *a*, odd from *b*."""
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

        pool: set[str] = set()

        def _add(s: str) -> None:
            s = " ".join(s.split())
            if s and s != a and s != b:
                pool.add(s)

        # 1. Base + every prefix-truncation of each unique word
        if base:
            _add(base)
        for w in ua + ub:
            for k in range(1, len(w)):
                _add(f"{base} {w[:k]}")

        # 2. Pair unique words -> common-prefix extensions / blend / mid-cut
        for wa_ in ua:
            for wb_ in ub:
                cp = cls._cpfx(wa_, wb_)
                if cp:
                    _add(f"{base} {cp}")
                    for e in range(1, max(len(wa_), len(wb_)) - len(cp) + 1):
                        for w in (wa_, wb_):
                            if len(cp) + e <= len(w):
                                _add(f"{base} {w[: len(cp) + e]}")
                _add(f"{base} {cls._blend(wa_, wb_)}")
                mid = (len(wa_) + len(wb_)) // 2
                if mid:
                    _add(f"{base} {wa_[:mid]}")
                    _add(f"{base} {wb_[:mid]}")

        # 3. Cross-structural mixing (comma-separated hybrids)
        if ua and ub:
            _add(f"{', '.join(ua)}, {b}")
            _add(f"{', '.join(ub)}, {a}")
            for x in ua[:2]:
                for y in ub[:2]:
                    _add(f"{x}, {' '.join(common + [y])}")
                    _add(f"{y}, {' '.join(common + [x])}")

        # 4. Word substitution inside each input's frame
        for i, w in enumerate(wa):
            for ww in wb:
                if w.lower() != ww.lower():
                    _add(" ".join(wa[:i] + [ww] + wa[i + 1 :]))
        for i, w in enumerate(wb):
            for ww in wa:
                if w.lower() != ww.lower():
                    _add(" ".join(wb[:i] + [ww] + wb[i + 1 :]))

        # 5. Subsets & (small) permutations of the word union
        union = list(dict.fromkeys(wa + wb))
        cap = min(len(union) + 1, 5)
        for r in range(1, cap):
            for c in combinations(union, r):
                _add(" ".join(c))
            if len(union) <= 6:
                for p in permutations(union, r):
                    _add(" ".join(p))

        # 6. Domain bridge terms
        for t in bridge:
            if base:
                _add(f"{base} {t}")
            for w in (ua + ub)[:2]:
                _add(f"{w} {t}")

        # 7. Containment: gradual extension of the shorter string
        lo_a, lo_b = a.lower(), b.lower()
        if lo_a.startswith(lo_b) or lo_b.startswith(lo_a):
            longer, shorter = (a, b) if len(a) >= len(b) else (b, a)
            extra = longer[len(shorter) :].strip()
            for k in range(1, len(extra) + 1):
                _add(f"{shorter} {extra[:k]}")

        return pool

    @classmethod
    def _generate_ambiguous_text(
        cls,
        val_a: str,
        val_b: str,
        n: int = 5,
        bridge_terms: list[str] | None = None,
    ) -> list[tuple[str, ...]]:
        val_words = list(
            dict.fromkeys(w for w in (val_a.split() + val_b.split()) if len(w) > 2)
        )
        bt = list(bridge_terms) if bridge_terms is not None else []
        bt = list(dict.fromkeys(bt + val_words))

        pool = cls._candidates(val_a, val_b, bt)
        ranked = sorted(
            ((c, *cls._score(c, val_a, val_b)) for c in pool),
            key=lambda r: r[1],
            reverse=True,
        )

        if len(ranked) < n:
            seen = {r[0] for r in ranked}
            fallbacks = [
                (tok, 0.0, 0.0, 0.0)
                for tok in val_words
                if tok not in seen and tok.lower() not in (val_a.lower(), val_b.lower())
            ]
            # If still not enough, use val_a and val_b themselves as a last resort.
            if not fallbacks:
                for fallback_str in (val_a, val_b):
                    if fallback_str not in seen:
                        fallbacks.append((fallback_str, 0.0, 0.0, 0.0))
                        seen.add(fallback_str)
            ranked = ranked + fallbacks[: n - len(ranked)]

        return ranked[:n]

    def _generate_ambiguous_references(self, cluster_reps, closest_correspondents):
        max_id = max(r.id.label for r in self._id_table)
        start_refs = {k: self._id_table.get(v) for k, v in cluster_reps.items()}

        ambiguous_refs = {}
        for cluster_id, start_ref in start_refs.items():
            self._progress.update(
                self._main_task,
                description=f"generating ambiguous cluster [{cluster_id}] reference",
            )
            if cluster_id not in closest_correspondents:
                self._progress.advance(self._main_task)
                continue
            target_cluster_id, _score, target = closest_correspondents[cluster_id]
            source_dict = start_ref.as_dict()
            target_dict = target.as_dict()
            all_keys = set(source_dict.keys()) | set(target_dict.keys())
            ref_properties = {
                "source": start_ref,
                "target": target,
                "target_cluster_id": target_cluster_id,
            }
            for k in all_keys:
                # attribute in one but not the other -> continue
                if (k in source_dict) ^ (k in target_dict):
                    continue
                val_a = start_ref[k]
                val_b = target[k]
                if val_a == val_b:
                    ref_properties[k] = val_a
                elif isinstance(val_a, str) and isinstance(val_b, str):
                    if k not in self._target_properties:
                        ref_properties[k] = ""
                    else:
                        results = self._generate_ambiguous_text(
                            val_a, val_b, 1, self._bridge_terms
                        )
                        ref_properties[k] = results[0][0] if results else val_a
                else:
                    ref_properties[k] = None
            new_ref = EntityReference(
                RefId(max_id + 1, "ambiguity-generator"), ref_properties
            )
            self._id_table.put(new_ref)
            ambiguous_refs[cluster_id] = new_ref.id
            max_id += 1
            self._progress.advance(self._main_task)
        return ambiguous_refs

    def best_pairing_scores(self) -> list[float]:
        """Best ambiguity score per cluster, independent of the threshold."""
        cluster_reps = self._get_cluster_representatives()
        saved = self._min_ambiguity
        self._min_ambiguity = None  # disable filtering: keep every pairing
        pairings = self._find_closest_rep(cluster_reps)
        self._min_ambiguity = saved
        return [score for _, score, _ in pairings.values()]

    def __call__(self) -> tuple[
        IdTable,
        dict[int, set[RefId]],
        dict[tuple[RefId, RefId], int],
        dict[tuple[RefId, RefId], int],
    ]:
        self._progress.start()
        cluster_reps = self._get_cluster_representatives()
        closest_reps = self._find_closest_rep(cluster_reps)
        ambiguous_refs = self._generate_ambiguous_references(cluster_reps, closest_reps)

        cluster_gt: dict[int, set[RefId]] = self._cluster_id_map.copy()
        for cluster_id, ref in ambiguous_refs.items():
            cluster_gt[cluster_id].add(ref)

        directed_gt: dict[ComparisonData, int] = {}
        undirected_gt: dict[ComparisonData, int] = {}
        for cluster_id, ref_ids in cluster_gt.items():
            for cmp_data in combinations(ref_ids, 2):
                undirected_gt[cmp_data] = 1
                directed_gt[cmp_data] = 1
            for ref in self._id_table.get_all(ref_ids):
                if not hasattr(ref, "target_cluster_id"):
                    continue
                fwd_comparison = (ref.id, ref.target.id)
                rev_comparison = (ref.target.id, ref.id)
                undirected_gt[fwd_comparison] = 1
                directed_gt[rev_comparison] = 2

        return self._id_table, cluster_gt, directed_gt, undirected_gt


def _compute_drops(bridge_counts: list[float]) -> list[tuple[int, int, float]]:
    """Return (prev_idx, curr_idx, drop_magnitude) for every consecutive fall."""
    return [
        (i - 1, i, bridge_counts[i - 1] - bridge_counts[i])
        for i in range(1, len(bridge_counts))
        if bridge_counts[i] < bridge_counts[i - 1]
    ]


def _find_gmm_highlight(df: pl.DataFrame, fit: GmmFit) -> dict[str, dict]:
    """Locate the bridge count at the GMM threshold on the sweep frame."""
    gmm_theta = fit.threshold
    thresholds = df["threshold"].to_list()
    counts = df["bridge_count"].to_list()

    closest_idx = int(np.argmin([abs(t - gmm_theta) for t in thresholds]))

    return {
        "gmm": {
            "threshold": gmm_theta,
            "bridge_count": counts[closest_idx],
            "extra": (
                fit.boundary_type if fit.boundary_type != "bayes_optimal" else None
            ),
        }
    }


def _plot_dataset_line(ax: plt.Axes, df: pl.DataFrame, color: str, label: str) -> None:
    thresholds = df["threshold"].to_list()
    counts = df["bridge_count"].to_list()

    ax.plot(thresholds, counts, color=color, linewidth=2, alpha=0.5, label=label)


_HIGHLIGHT_STYLES: dict[str, dict] = {
    "gmm": {"marker": "D", "label": "GMM threshold"},
}


def _build_highlight_legend(ax: plt.Axes) -> None:
    """Append criterion-shape entries to the existing legend."""
    shape_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=style["marker"],
            color="grey",
            linestyle="None",
            markersize=7,
            label=style["label"],
        )
        for style in _HIGHLIGHT_STYLES.values()
    ]
    existing_handles, existing_labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=existing_handles + shape_handles,
        labels=existing_labels + [s["label"] for s in _HIGHLIGHT_STYLES.values()],
        title="Dataset / threshold",
        frameon=True,
        fontsize=9,
    )


def _plot_threshold_highlights(
    ax: plt.Axes,
    highlights: dict[str, dict],
    color: str,
    y_max: float,
    dataset_idx: int,
) -> None:
    """Draw vertical lines, axis markers, and threshold+bridge-count labels."""
    for criterion_idx, (criterion, style) in enumerate(_HIGHLIGHT_STYLES.items()):
        info = highlights[criterion]
        x = float(info["threshold"])

        ax.axvline(x=x, color=color, linestyle="--", linewidth=0.8, alpha=0.6)

        # Separate markers vertically: one row per criterion.
        marker_y = -(0.06 + criterion_idx * 0.05) * y_max
        ax.plot(
            x,
            marker_y,
            marker=style["marker"],
            color=color,
            markersize=7,
            clip_on=False,
            zorder=5,
        )

        label_parts = [f"θ={x:.2f}", f"b={info['bridge_count']}"]
        if info["extra"]:
            label_parts.append(f"({info['extra']})")
        label = "\n".join(label_parts)

        # Stack by both dataset and criterion to avoid any overlap.
        n_criteria = len(_HIGHLIGHT_STYLES)
        label_y = -(0.09 + (dataset_idx * n_criteria + criterion_idx) * 0.07) * y_max
        ax.annotate(
            label,
            xy=(x, 0),
            xytext=(x, label_y),
            color=color,
            fontsize=7,
            ha="center",
            va="top",
            clip_on=False,
            annotation_clip=False,
        )


def _render_threshold_plot(
    all_frames: list[pl.DataFrame],
    all_fits: list[GmmFit],
    all_scores: list[list[float]],
    out_path: Path,
) -> None:
    """Render the threshold-vs-bridge-count sweep as a 1×N faceted grid, one panel
    per dataset with its own x/y scale and the x-axis cropped to the action region
    ([min(scores)−1, max(scores)+1]). The GMM threshold is marked per panel.
    Supplemental artifact — matplotlib titles carry the dataset name + boundary type."""
    sns.set_theme(style="white", font_scale=1.0)
    n = len(all_frames)
    _fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, df, fit, scores in zip(axes, all_frames, all_fits, all_scores):
        dataset_name = df["dataset"][0]
        thresholds = df["threshold"].to_list()
        counts = df["bridge_count"].to_list()

        ax.plot(thresholds, counts, color="#1f77b4", linewidth=1.8)
        ax.axvline(x=fit.threshold, color="black", linestyle="--", linewidth=1.0)

        # Crop x-axis to the action region.
        scores_arr = np.array(scores, dtype=float)
        x_lo = float(scores_arr.min()) - 1.0
        x_hi = float(scores_arr.max()) + 1.0
        ax.set_xlim(x_lo, x_hi)

        # Annotation: θ and bridge count at θ, placed at the top.
        bridge_at_theta = sum(1 for s in scores if s >= fit.threshold)
        ann = f"θ={fit.threshold:.2f}\nb={bridge_at_theta}"
        ax.annotate(
            ann,
            xy=(fit.threshold, counts[0] if counts else 0),
            xytext=(0.5, 0.92),
            textcoords="axes fraction",
            ha="center",
            fontsize=7,
            color="black",
        )

        ax.set_title(f"{dataset_name} ({fit.boundary_type})", fontsize=9)
        ax.set_xlabel("θ", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("Bridge count", fontsize=8)
        ax.tick_params(labelsize=7)
        sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _render_gmm_fit_plot(
    scores: list[float], fit: GmmFit, dataset_name: str, out_path: Path
) -> None:
    """Render the histogram of best-pairing scores with the two fitted Gaussian
    density curves and the threshold marked, one figure per dataset."""
    sns.set_theme(style="white", font_scale=1.1)
    _fig, ax = plt.subplots(figsize=(7, 4))

    scores_arr = np.array(scores, dtype=float)
    lo = float(min(scores_arr.min(), fit.mu_superficial - 3 * fit.sigma_superficial))
    hi = float(max(scores_arr.max(), fit.mu_genuine + 3 * fit.sigma_genuine))
    x = np.linspace(lo, hi, 400)

    ax.hist(
        scores_arr,
        bins="auto",
        density=True,
        color="lightgrey",
        edgecolor="white",
        label="best-pairing scores",
    )

    genuine_density = fit.pi_genuine * norm.pdf(x, fit.mu_genuine, fit.sigma_genuine)
    superficial_density = fit.pi_superficial * norm.pdf(
        x, fit.mu_superficial, fit.sigma_superficial
    )

    ax.plot(x, genuine_density, color="#2ca02c", linewidth=2, label="genuine overlap")
    ax.plot(
        x,
        superficial_density,
        color="#d62728",
        linewidth=2,
        label="superficial overlap",
    )
    ax.axvline(
        x=fit.threshold,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"θ={fit.threshold:.2f}",
    )

    ax.set_xlabel("Ambiguity score")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, frameon=True)
    sns.despine()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _write_methodology_csv(rows: list[dict], out_path: Path) -> None:
    """Write the per-dataset methodology summary CSV used to populate the paper's
    ambiguity-injection table."""
    df = pl.DataFrame(rows)
    df.write_csv(out_path, include_header=True)


def _fit_gmm_threshold(scores: list[float], n_init: int = 50) -> GmmFit:
    """Fit a two-component Gaussian Mixture Model to ``scores`` and return the
    Bayes-optimal decision boundary together with the fitted component parameters.

    The scores are modelled as a mixture of two populations: *genuine* overlap
    (high-mean component, pairs whose representatives meaningfully resemble each
    other) and *superficial* overlap (low-mean component, pairs whose resemblance
    is only on the surface). The threshold is the score at which the posterior
    probability of belonging to the genuine component equals 0.5.

    When the two components are not separable (the posterior never reaches 0.5
    between the means), the threshold falls back to the midpoint of the two means
    and ``boundary_type`` is set to ``"mean_midpoint"`` so the degenerate fit is
    reportable.

    Multiple random restarts (``n_init``) are used; ``GaussianMixture`` keeps the
    run with the highest log-likelihood automatically.
    """
    X = np.array(scores).reshape(-1, 1)

    gmm = GaussianMixture(
        n_components=2, covariance_type="full", n_init=n_init, random_state=0
    )
    gmm.fit(X)

    means = gmm.means_.flatten()
    high_idx = int(np.argmax(means))  # genuine = higher mean
    low_idx = 1 - high_idx  # superficial = lower mean

    w_genuine = gmm.weights_[high_idx]
    mu_genuine = float(means[high_idx])
    sigma_genuine = float(np.sqrt(gmm.covariances_[high_idx, 0, 0]))

    w_superficial = gmm.weights_[low_idx]
    mu_superficial = float(means[low_idx])
    sigma_superficial = float(np.sqrt(gmm.covariances_[low_idx, 0, 0]))

    def posterior_diff(x):
        """P(genuine | x) - 0.5, zero when the two components are equally likely."""
        p_genuine = w_genuine * norm.pdf(x, mu_genuine, sigma_genuine)
        p_superficial = w_superficial * norm.pdf(x, mu_superficial, sigma_superficial)
        total = p_genuine + p_superficial
        if total == 0:
            return -0.5
        return (p_genuine / total) - 0.5

    lo, hi = min(mu_superficial, mu_genuine), max(mu_superficial, mu_genuine)
    try:
        threshold = float(brentq(posterior_diff, lo, hi))
        boundary_type = "bayes_optimal"
    except ValueError:
        threshold = (mu_superficial + mu_genuine) / 2.0
        boundary_type = "mean_midpoint"

    return GmmFit(
        threshold=threshold,
        mu_genuine=mu_genuine,
        sigma_genuine=sigma_genuine,
        pi_genuine=float(w_genuine),
        mu_superficial=mu_superficial,
        sigma_superficial=sigma_superficial,
        pi_superficial=float(w_superficial),
        boundary_type=boundary_type,
    )


def _analyse_optimum_threshold(
    analysis_start_value: float,
    analysis_step_count: float,
    analysis_step_size: float,
    data: CsvBenchmarkData,
    introduce_ambiguity: AmbiguityGenerator,
    gmm_restarts: int = 50,
) -> tuple[pl.DataFrame, GmmFit, list[float]]:
    scores = introduce_ambiguity.best_pairing_scores()
    threshold_range = np.arange(
        analysis_start_value,
        analysis_step_size * (analysis_step_count + 1),
        analysis_step_size,
    )
    gmm_fit = _fit_gmm_threshold(scores, n_init=gmm_restarts)

    display_name = {
        "affiliationstrings": "affiliations",
        "cora1": "cora",
        "fodors_zagat_nophone": "fodors-zagat-nophone",
        "geographicalSettelments": "geographic-settlements",
    }.get(data.name, data.name)
    cluster_count = len(data.compute_clusters())

    return (
        pl.DataFrame(
            data=[
                {
                    "dataset": display_name,
                    "threshold": t,
                    "bridge_count": sum(1 for s in scores if s >= t),
                    "cluster_count": cluster_count,
                }
                for t in threshold_range
            ]
        ),
        gmm_fit,
        scores,
    )


def _write_analysis(
    analysis_frames: list[DataFrame],
    analysis_fits: list[GmmFit],
    analysis_scores: list[list[float | int]],
    methodology_rows: list[dict[Any, Any]],
    plot_dir: Path,
) -> None:
    """Write out the ambiguity analysis of the data frames."""
    if not analysis_frames:
        return

    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    for df, fit, scores in zip(analysis_frames, analysis_fits, analysis_scores):
        dataset_name = df["dataset"][0]
        bell_path = plot_path / f"gmm_fit_{dataset_name}.png"
        _render_gmm_fit_plot(scores, fit, dataset_name, bell_path)

        bridge_count_at_theta = sum(1 for s in scores if s >= fit.threshold)
        methodology_rows.append(
            {
                "dataset": dataset_name,
                "cluster_count": df["cluster_count"][0],
                "bridge_count_at_theta": bridge_count_at_theta,
                "bridge_count_per_cluster": bridge_count_at_theta
                / df["cluster_count"][0],
                "theta": fit.threshold,
                "mu_genuine": fit.mu_genuine,
                "sigma_genuine": fit.sigma_genuine,
                "pi_genuine": fit.pi_genuine,
                "mu_superficial": fit.mu_superficial,
                "sigma_superficial": fit.sigma_superficial,
                "pi_superficial": fit.pi_superficial,
                "boundary_type": fit.boundary_type,
            }
        )

    _write_methodology_csv(methodology_rows, plot_path / "ambiguity_methodology.csv")

    # Write the full threshold sweep (structural theta-sensitivity) for the
    # supplemental data package.
    pl.concat(analysis_frames).write_csv(
        plot_path / "ambiguity_threshold_sweep.csv", include_header=True
    )

    _render_threshold_plot(
        analysis_frames,
        analysis_fits,
        analysis_scores,
        plot_path / "threshold_analysis.png",
    )


def _generate_amb_datasets(
    data: CsvBenchmarkData, ambiguity_generator: AmbiguityGenerator, output_dir: Path
):
    """Generate two datasets containing ambiguous bridge records."""
    id_table, cluster_gt, directed_gt, undirected_gt = ambiguity_generator()
    table_ = [
        {
            "id": ref.id.label,
            "source": str(ref.id.source),
            **{
                k: v
                for k, v in ref.as_dict().items()
                if k not in {"source", "target", "target_cluster_id"}
            },
        }
        for ref in id_table
    ]
    data_df = pl.DataFrame(table_)
    clusters_df = pl.DataFrame(
        [
            {"id": ref_id.label, "source": ref_id.source, "cluster_id": cluster_id}
            for cluster_id, ref_ids in cluster_gt.items()
            for ref_id in ref_ids
        ]
    )
    directed_df = pl.DataFrame(
        [
            {
                "left_id": left_id.label,
                "left_source": left_id.source,
                "right_id": right_id.label,
                "right_source": right_id.source,
                "label": c,
            }
            for (left_id, right_id), c in directed_gt.items()
        ]
    )
    undirected_df = pl.DataFrame(
        [
            {
                "left_id": left_id.label,
                "left_source": left_id.source,
                "right_id": right_id.label,
                "right_source": right_id.source,
                "label": c,
            }
            for (left_id, right_id), c in undirected_gt.items()
        ]
    )
    directed_undirected = {
        f"amb_{data.name}": undirected_df,
        f"amb_{data.name}_dir": directed_df,
    }

    for output_dataset_name, mapping_df in directed_undirected.items():
        dataset_output_path = output_dir / output_dataset_name
        dataset_output_path.mkdir(parents=True, exist_ok=True)
        file_map = {
            f"{output_dataset_name}_ids.csv": data_df,
            f"{output_dataset_name}_mapping.csv": mapping_df,
            f"{output_dataset_name}_cluster_mapping.csv": clusters_df,
        }
        for name, df in file_map.items():
            df.write_csv(dataset_output_path / name, include_header=True)


@matchescu.command("ambiguity-generator")
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
    "--analyse-best-threshold",
    "is_analysis_mode",
    type=click.BOOL,
    default=False,
    required=False,
    is_flag=True,
    help="Run one-factor-at-a-time analysis for the best ambiguity threshold",
)
@click.option(
    "--threshold-analysis-step-count",
    "analysis_step_count",
    type=click.FLOAT,
    default=10.0,
    required=False,
    help="How many thresholds to analyse using one-factor-at-a-time",
)
@click.option(
    "--threshold-analysis-step-size",
    "analysis_step_size",
    type=click.FLOAT,
    default=1.0,
    required=False,
    help="How large should the difference between consecutive thresholds be for one-factor-at-a-time analysis",
)
@click.option(
    "--threshold-analysis-start-value",
    "analysis_start_value",
    type=click.FLOAT,
    default=0.0,
    required=False,
    help="How large should the difference between consecutive thresholds be for one-factor-at-a-time analysis",
)
@click.option(
    "--gmm-restarts",
    "gmm_restarts",
    type=click.INT,
    default=50,
    required=False,
    help="Number of random restarts for the Gaussian mixture model fit",
)
@click.option(
    "--plot-dir",
    "plot_dir",
    type=click.Path(
        exists=False, file_okay=False, dir_okay=True, writable=True, resolve_path=True
    ),
    default=".",
    required=False,
    help="Directory where plots and the methodology CSV are written",
)
@click.pass_context
def main(
    ctx: Context,
    config_file: str | PathLike,
    output_dir: str | PathLike,
    is_analysis_mode: bool,
    analysis_step_count: float,
    analysis_step_size: float,
    analysis_start_value: float,
    gmm_restarts: int,
    plot_dir: str | PathLike,
):
    """Introduce ambiguous data into existing dataset."""
    root_dir = get_options(ctx).root_dir
    config: AmbiguityConfig = (
        JSONConfig(make_absolute_path(config_file, root_dir), AmbiguityConfig)
        .load()
        .config_obj
    )
    output_dir = make_absolute_path(output_dir, root_dir)

    analysis_frames: list[pl.DataFrame] = []
    analysis_fits: list[GmmFit] = []
    analysis_scores: list[list[float]] = []
    methodology_rows: list[dict] = []
    for ds_idx, input_cfg in enumerate(config.input_data):
        print("processing", input_cfg.dataset.directory)
        builder = new_benchmark_data_factory(input_cfg.dataset, root_dir)
        data = builder.load_data().create()
        amb_generator = AmbiguityGenerator(
            data.id_table,
            data.true_matches,
            data.compute_clusters(),
            input_cfg.ambiguity_targets,
            min_ambiguity=input_cfg.min_ambiguity,
        )

        # in analysis mode, only compute the optimal cutoff point at which
        # realistic bridge records would be hard to generate and move on
        if is_analysis_mode:
            sweep_df, gmm_fit, scores = _analyse_optimum_threshold(
                analysis_start_value,
                analysis_step_count,
                analysis_step_size,
                data,
                amb_generator,
                gmm_restarts=gmm_restarts,
            )
            analysis_frames.append(sweep_df)
            analysis_fits.append(gmm_fit)
            analysis_scores.append(scores)
        else:
            _generate_amb_datasets(data, amb_generator, output_dir)

    _write_analysis(
        analysis_frames, analysis_fits, analysis_scores, methodology_rows, plot_dir
    )
