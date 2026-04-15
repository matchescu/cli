import re

from difflib import SequenceMatcher
from itertools import combinations, permutations
from os import PathLike
from pathlib import Path
from typing import Set

import click
import networkx as nx
import polars as pl
from click import Context
from rich.progress import Progress

from matchescu.cli._cmd_group import matchescu
from matchescu.cli.config import JSONConfig, AmbiguityConfig, new_benchmark_data_factory
from matchescu.cli.runtime import get_options, make_absolute_path
from matchescu.reference_store.id_table import IdTable
from matchescu.typing import (
    EntityReferenceIdentifier as RefId,
    EntityReference,
)

try:
    from rapidfuzz.distance import Levenshtein
except ImportError:
    try:
        import Levenshtein
    except ImportError:
        Levenshtein = None


DIR = Path("./data/affiliationstrings/")
DEFAULT_BRIDGE_TERMS: list[str] = [
    "Research",
    "Center",
    "Lab",
    "Institute",
    "College",
    "Department",
    "Division",
    "Group",
    "School",
    "Foundation",
]


type ComparisonData = tuple[RefId, RefId]


class AmbiguityGenerator:
    _DEFAULT_DEGRADATION_PATTERNS = [
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
        ambiguity_target_properties: list[str] = None,
        string_degradation_patterns: list[str] = None,
        min_ambiguity: float | None = None,
    ) -> None:
        self._mapping_gt = mapping_gt
        self._id_table = id_table
        self._cluster_id_map, self._cluster_count = self.__get_clusters()
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

    def __get_clusters(self):
        ref_ids = list(self._mapping_gt)
        g = nx.DiGraph(ref_ids)
        cluster_count = 0
        cluster_ref_map = {}

        for cluster_count, cluster in enumerate(nx.strongly_connected_components(g), 1):
            if cluster_count not in cluster_ref_map:
                cluster_ref_map[cluster_count] = set()
            for ref_id in cluster:
                cluster_ref_map[cluster_count].add(ref_id)

        return cluster_ref_map, cluster_count

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
        - property does not exist in ``x`` or ``y`` other gets -2.0

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
            candidates = list(ref_ids)
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
        bt = bridge_terms if bridge_terms is not None else DEFAULT_BRIDGE_TERMS
        pool = cls._candidates(val_a, val_b, bt)
        ranked = sorted(
            ((c, *cls._score(c, val_a, val_b)) for c in pool),
            key=lambda r: r[1],
            reverse=True,
        )
        return ranked[:n]

    def _generate_ambiguous_references(self, cluster_reps, closest_correspondents):
        max_id = max(map(lambda r: r.id.label, self._id_table))
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
            target_cluster_id, score, target = closest_correspondents[cluster_id]
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
                        ref_properties[k] = self._generate_ambiguous_text(
                            val_a, val_b, 1
                        )[0][0]
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

    def __call__(self) -> tuple[
        IdTable,
        dict[int, Set[RefId]],
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
@click.pass_context
def main(ctx: Context, config_file: str | PathLike, output_dir: str | PathLike):
    """Introduce ambiguous data into existing dataset."""
    root_dir = get_options(ctx).root_dir
    config: AmbiguityConfig = (
        JSONConfig(make_absolute_path(config_file, root_dir), AmbiguityConfig)
        .load()
        .config_obj
    )
    output_dir = make_absolute_path(output_dir, root_dir)

    for ds_idx, input_cfg in enumerate(config.input_data):
        print("processing", input_cfg.dataset.directory)
        builder = new_benchmark_data_factory(input_cfg.dataset, root_dir)
        data = builder.load_data().create()
        introduce_ambiguity = AmbiguityGenerator(
            data.id_table,
            data.true_matches,
            input_cfg.ambiguity_targets,
            min_ambiguity=input_cfg.min_ambiguity,
        )
        id_table, cluster_gt, directed_gt, undirected_gt = introduce_ambiguity()
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
