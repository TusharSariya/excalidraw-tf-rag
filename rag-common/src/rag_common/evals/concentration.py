"""Reference-concentration metric for clustering quality (D10/D13 — labels-free eval
protocol for landscape mapping, cf. Boyack & Klavans).

For a gold survey (a doc known to reference >=100 in-corpus papers, or a section of one),
a *good* topical partition should place most of the survey's references in a small number
of clusters — a survey mostly "about" retrieval-augmented generation should have most of
its refs land in one or two RAG-ish clusters, not scattered uniformly across every cluster
in the corpus.

**Cluster-size-aware (D10 hardening item):** raw concentration is meaningless on its own —
a partition with one giant cluster covering 90% of the corpus trivially "concentrates"
every survey's refs without encoding anything about topical coherence. So every observed
score is compared against two null distributions built from the *same* partition:

- uniform-random: each doc independently assigned to one of the partition's cluster ids
  uniformly at random (ignores the real size distribution entirely).
- size-matched-random: the same per-cluster sizes as the real partition, but doc-to-cluster
  membership shuffled (the "the big-cluster trivially wins" confound, controlled for).

Both nulls are drawn `n=100` times; the report carries each null's mean/std and the
observed score's z-score against it. `z_size_matched` is the metric to gate a promotion
decision on — it's the one that isn't fooled by cluster-size skew alone.
"""

from __future__ import annotations

import random
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = 1


def herfindahl(cluster_ids: list[int]) -> float:
    """Herfindahl-Hirschman concentration index over a list of cluster assignments.

    ``sum(p_i^2)`` where ``p_i`` is the fraction of items in cluster ``i``. Ranges from
    ``1/k`` (uniform spread across ``k`` distinct touched clusters) to ``1.0`` (every item
    in the same cluster — perfect concentration). Empty input returns ``0.0``.
    """
    n = len(cluster_ids)
    if n == 0:
        return 0.0
    counts = Counter(cluster_ids)
    return sum((c / n) ** 2 for c in counts.values())


def score_refs(
    ref_doc_ids: list[str],
    partition: dict[str, int],
) -> float | None:
    """Concentration score for one survey's (or section's) reference list against a
    partition. Refs not present in ``partition`` are dropped (un-clustered docs, e.g.
    metadata-only or filtered out of the doc-vector index). Returns ``None`` when fewer
    than 2 refs are placeable — concentration is undefined/trivial below that."""
    placed = [partition[d] for d in ref_doc_ids if d in partition]
    if len(placed) < 2:
        return None
    return herfindahl(placed)


def uniform_random_partition(
    doc_ids: list[str],
    cluster_ids: list[int],
    *,
    rng: random.Random,
) -> dict[str, int]:
    """Assign every doc in ``doc_ids`` to a uniformly random choice from ``cluster_ids``
    (with replacement) — ignores the real partition's size distribution entirely."""
    return {d: rng.choice(cluster_ids) for d in doc_ids}


def size_matched_random_partition(
    partition: dict[str, int],
    *,
    rng: random.Random,
) -> dict[str, int]:
    """Shuffle membership across the same docs while preserving the exact per-cluster
    size distribution of ``partition`` (a permutation of its values)."""
    doc_ids = list(partition)
    values = [partition[d] for d in doc_ids]
    shuffled = list(values)
    rng.shuffle(shuffled)
    return dict(zip(doc_ids, shuffled))


@dataclass
class NullStats:
    mean: float
    std: float
    z: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": round(self.mean, 6),
            "std": round(self.std, 6),
            "z": round(self.z, 4) if self.z is not None else None,
        }


def _null_stats(observed: float, draws: list[float]) -> NullStats:
    mean = statistics.fmean(draws) if draws else 0.0
    std = statistics.pstdev(draws) if len(draws) > 1 else 0.0
    z = (observed - mean) / std if std > 0 else None
    return NullStats(mean=mean, std=std, z=z)


@dataclass
class SurveyScore:
    survey_doc_id: str
    section_scores: dict[str, float] = field(default_factory=dict)
    mean_score: float | None = None
    uniform_null: NullStats | None = None
    size_matched_null: NullStats | None = None
    ref_count: int = 0
    placed_ref_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "survey_doc_id": self.survey_doc_id,
            "section_scores": {k: round(v, 6) for k, v in self.section_scores.items()},
            "mean_score": round(self.mean_score, 6) if self.mean_score is not None else None,
            "uniform_null": self.uniform_null.to_dict() if self.uniform_null else None,
            "size_matched_null": self.size_matched_null.to_dict()
            if self.size_matched_null
            else None,
            "ref_count": self.ref_count,
            "placed_ref_count": self.placed_ref_count,
        }


def score_survey(
    survey_doc_id: str,
    sections: dict[str, list[str]],
    partition: dict[str, int],
    *,
    n_random: int = 100,
    seed: int = 42,
) -> SurveyScore | None:
    """Score one survey (possibly multi-section) against a partition, with both null
    distributions. Returns ``None`` if no section has >=2 placeable refs (nothing to
    score for this survey against this partition/corpus)."""
    rng = random.Random(seed)
    cluster_ids = sorted(set(partition.values()))
    doc_ids = list(partition)
    all_refs = sorted({d for refs in sections.values() for d in refs})

    section_scores: dict[str, float] = {}
    for section, refs in sections.items():
        score = score_refs(refs, partition)
        if score is not None:
            section_scores[section] = score

    if not section_scores:
        return None

    mean_score = statistics.fmean(section_scores.values())

    # Nulls are drawn once per survey (pooled across sections via the same combined ref
    # list) rather than per section — cheaper, and the null question ("would *these*
    # docs concentrate this much under a random partition?") doesn't depend on how the
    # refs are split into sections.
    uniform_draws = []
    size_matched_draws = []
    for _ in range(n_random):
        rand_partition = uniform_random_partition(doc_ids, cluster_ids, rng=rng)
        section_vals = [score_refs(refs, rand_partition) for refs in sections.values()]
        vals = [v for v in section_vals if v is not None]
        if vals:
            uniform_draws.append(statistics.fmean(vals))

        rand_partition = size_matched_random_partition(partition, rng=rng)
        section_vals = [score_refs(refs, rand_partition) for refs in sections.values()]
        vals = [v for v in section_vals if v is not None]
        if vals:
            size_matched_draws.append(statistics.fmean(vals))

    placed = len({d for d in all_refs if d in partition})
    return SurveyScore(
        survey_doc_id=survey_doc_id,
        section_scores=section_scores,
        mean_score=mean_score,
        uniform_null=_null_stats(mean_score, uniform_draws) if uniform_draws else None,
        size_matched_null=_null_stats(mean_score, size_matched_draws)
        if size_matched_draws
        else None,
        ref_count=len(all_refs),
        placed_ref_count=placed,
    )


@dataclass
class ConcentrationReport:
    surveys: list[SurveyScore]
    mean_score: float | None
    mean_z_uniform: float | None
    mean_z_size_matched: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "survey_count": len(self.surveys),
            "mean_score": round(self.mean_score, 6) if self.mean_score is not None else None,
            "mean_z_uniform": round(self.mean_z_uniform, 4)
            if self.mean_z_uniform is not None
            else None,
            "mean_z_size_matched": round(self.mean_z_size_matched, 4)
            if self.mean_z_size_matched is not None
            else None,
            "surveys": [s.to_dict() for s in self.surveys],
        }


def concentration_report(
    gold: dict[str, dict[str, list[str]]],
    partition: dict[str, int],
    *,
    n_random: int = 100,
    seed: int = 42,
) -> ConcentrationReport:
    """Score every gold survey's references against ``partition`` (``doc_id -> cluster
    id``), with both null comparisons.

    ``gold`` is ``{survey_doc_id: {section_path: [ref_doc_id, ...]}}`` — pass
    ``{"_all": [...]}`` as the sole section for doc-level (non-sectioned) gold.
    """
    scores: list[SurveyScore] = []
    for i, (survey_doc_id, sections) in enumerate(sorted(gold.items())):
        result = score_survey(survey_doc_id, sections, partition, n_random=n_random, seed=seed + i)
        if result is not None:
            scores.append(result)

    mean_score = (
        statistics.fmean(s.mean_score for s in scores if s.mean_score is not None)
        if scores
        else None
    )
    z_uniform = [
        s.uniform_null.z for s in scores if s.uniform_null and s.uniform_null.z is not None
    ]
    z_size_matched = [
        s.size_matched_null.z
        for s in scores
        if s.size_matched_null and s.size_matched_null.z is not None
    ]
    return ConcentrationReport(
        surveys=scores,
        mean_score=mean_score,
        mean_z_uniform=statistics.fmean(z_uniform) if z_uniform else None,
        mean_z_size_matched=statistics.fmean(z_size_matched) if z_size_matched else None,
    )
