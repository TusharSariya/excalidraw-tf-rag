"""Ablation harness: hybrid vs text-only vs citation-only clustering, scored by
reference-concentration, with a PROMOTE/TIE/BLOCK gate.

Reuses `rag_common.litmap` for every clustering step — `similarity.py` for per-signal kNN
+ RRF fusion, `communities.py` for Leiden partitioning — this module only orchestrates the
`alpha` settings and folds the results through `concentration.py`.

**Granularity-matched comparison (hardening pass, post-provisional-run).** The provisional
map-gate run found citation_only "winning" on GLR with 1,762 clusters vs hybrid's 15 — a
sparse/partly-unkeyed citation graph fragments into tiny (~size-2/3) clusters that trivially
concentrate survey refs even against a size-matched null, because the null preserves that
same tiny-cluster size distribution. Comparing arms at wildly different cluster counts is
not a valid comparison. So every arm is swept over Leiden `resolution` to find the operating
point whose >=2-doc cluster count is closest to a shared `target_k` (default: the hybrid
arm's natural alpha=0.5/resolution=1.0 cluster count), and only compared to other arms at
that matched granularity. `coverage`/`singleton_share` (see `concentration.cluster_coverage`)
make the pathology visible even when granularity is matched: an arm that refuses to cluster
most of the corpus (citation-only on unkeyed docs) still shows a low `coverage`, and
`restrict_to_common_support` prevents an arm from getting credit for surveys whose refs it
personally declined to place while another arm did place them — every arm is scored on the
same, common set of refs (covered by ALL arms compared).

**Promotion gate, paired-comparison based.** Per-survey score differences (hybrid minus
each other arm, over the common-support gold) get a sign test + a bootstrap CI on the mean
difference. `PROMOTE` only if hybrid beats *both* other arms with a CI that excludes 0 (and
is positive); `TIE` if a CI includes 0 (no significant difference either way); `BLOCK` if
any arm's CI excludes 0 in its favor (hybrid loses) — small survey counts (dozens, not
hundreds) make z-score-only comparisons overconfident, hence the CI/tie machinery.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from rag_common.evals.concentration import (
    ConcentrationReport,
    cluster_count_min2,
    cluster_coverage,
    concentration_report,
    score_refs,
    score_survey,
)
from rag_common.litmap.communities import leiden_communities
from rag_common.litmap.similarity import bib_coupling_all, cosine_knn_all, fuse_neighbors
from rag_common.litmap.types import CitationAdjacency

SCHEMA_VERSION = 2

# (label, alpha) — alpha=0 is pure text (cosine kNN only), alpha=1 is pure citation
# (bib-coupling only); both are exact reductions of `fuse_neighbors` (see similarity.py).
ABLATION_VARIANTS: tuple[tuple[str, float], ...] = (
    ("hybrid", 0.5),
    ("text_only", 0.0),
    ("citation_only", 1.0),
)

# `eval map-gate --alpha-sweep` — the D22/decisions-log-pre-approved alpha sweep. Kept as
# its own tuple (not folded into ABLATION_VARIANTS) so the default 3-arm gate run stays the
# cheap path; sweeping adds two more full resolution-sweep + concentration passes.
ALPHA_SWEEP_VARIANTS: tuple[tuple[str, float], ...] = (
    ("citation_only", 1.0),
    ("hybrid_a75", 0.75),
    ("hybrid", 0.5),
    ("hybrid_a25", 0.25),
    ("text_only", 0.0),
)

# Geometric-ish grid spanning very low resolutions (needed to merge an over-fragmented
# citation-coupling graph — the GLR provisional run hit 1,762 clusters at resolution=1.0 on
# 5,819 docs, two orders of magnitude above the hybrid arm's natural ~15) up through
# resolutions that under-merge a small/sparse text graph.
DEFAULT_RESOLUTION_GRID: tuple[float, ...] = (
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.35,
    0.5,
    0.75,
    1.0,
    1.5,
    2.2,
    3.5,
    5.0,
    8.0,
)


def build_fused_edges(
    doc_vectors: dict[str, list[float]],
    adjacency: CitationAdjacency | None,
    *,
    alpha: float,
    top_k: int = 10,
) -> list[tuple[str, str, float]]:
    """RRF-fused kNN edges for one alpha setting. Split out from `build_partition` so a
    resolution sweep (many Leiden runs at different `resolution`) doesn't repeat the O(n^2)
    cosine matmul / bib-coupling scan once per resolution step — those only depend on
    `alpha`, not `resolution`."""
    doc_ids = list(doc_vectors)
    effective_alpha = alpha if adjacency is not None else 0.0
    text_knn = cosine_knn_all(doc_vectors, top_k=top_k)
    cite_knn = (
        bib_coupling_all(doc_ids, adjacency, top_k=top_k)
        if adjacency is not None and effective_alpha > 0.0
        else {}
    )

    edge_scores: dict[tuple[str, str], float] = {}
    for doc_id in doc_ids:
        fused = fuse_neighbors(
            text_knn.get(doc_id, []),
            cite_knn.get(doc_id, []),
            alpha=effective_alpha,
            top_k=top_k,
        )
        for n in fused:
            key = (doc_id, n.doc_id) if doc_id < n.doc_id else (n.doc_id, doc_id)
            edge_scores[key] = max(edge_scores.get(key, 0.0), n.score)
    return [(u, v, w) for (u, v), w in edge_scores.items()]


def build_partition(
    doc_vectors: dict[str, list[float]],
    adjacency: CitationAdjacency | None,
    *,
    alpha: float,
    top_k: int = 10,
    seed: int = 42,
    resolution: float = 1.0,
) -> dict[str, int]:
    """One Leiden partition over the fused (or single-signal) kNN graph, `doc_id ->
    cluster id`. `adjacency=None` forces `alpha=0` regardless of the requested value —
    there is no citation signal to fuse (caller should skip citation_only/hybrid variants
    entirely in that case; kept as a hard fallback here so this never raises)."""
    edges = build_fused_edges(doc_vectors, adjacency, alpha=alpha, top_k=top_k)
    return leiden_communities(
        edges, all_doc_ids=list(doc_vectors), seed=seed, resolution=resolution
    )


@dataclass
class ResolutionPoint:
    """One step of a per-arm resolution sweep: granularity + a cheap (no null-draw) mean
    concentration score, so the full score-vs-K curve is visible in the report even though
    only the chosen (closest-to-`target_k`) operating point gets full null-controlled
    scoring."""

    resolution: float
    cluster_count: int
    cluster_count_min2: int
    coverage: float
    singleton_share: float
    mean_score: float | None
    survey_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolution": self.resolution,
            "cluster_count": self.cluster_count,
            "cluster_count_min2": self.cluster_count_min2,
            "coverage": round(self.coverage, 6),
            "singleton_share": round(self.singleton_share, 6),
            "mean_score": round(self.mean_score, 6) if self.mean_score is not None else None,
            "survey_count": self.survey_count,
        }


def sweep_resolution(
    doc_vectors: dict[str, list[float]],
    adjacency: CitationAdjacency | None,
    gold: dict[str, dict[str, list[str]]],
    *,
    alpha: float,
    target_k: int,
    top_k: int = 10,
    seed: int = 42,
    resolutions: tuple[float, ...] = DEFAULT_RESOLUTION_GRID,
) -> tuple[dict[str, int], list[ResolutionPoint]]:
    """Sweep Leiden `resolution` for one arm, return the partition whose >=2-doc cluster
    count is closest to `target_k`, plus the full score-vs-K curve (every resolution step)
    for granularity-sensitivity reporting. The curve's `mean_score` is unweighted raw mean
    concentration (n_random=0 — no null draws) purely to keep the sweep cheap; the chosen
    operating point is re-scored with full null stats by the caller."""
    effective_alpha = alpha if adjacency is not None else 0.0
    edges = build_fused_edges(doc_vectors, adjacency, alpha=effective_alpha, top_k=top_k)
    doc_ids = list(doc_vectors)

    points: list[ResolutionPoint] = []
    partitions_by_resolution: dict[float, dict[str, int]] = {}
    for r in resolutions:
        partition = leiden_communities(edges, all_doc_ids=doc_ids, seed=seed, resolution=r)
        partitions_by_resolution[r] = partition
        cov = cluster_coverage(partition, doc_ids)
        scores = []
        for survey_id, sections in sorted(gold.items()):
            result = score_survey(survey_id, sections, partition, n_random=0)
            if result is not None and result.mean_score is not None:
                scores.append(result.mean_score)
        points.append(
            ResolutionPoint(
                resolution=r,
                cluster_count=len(set(partition.values())),
                cluster_count_min2=cluster_count_min2(partition),
                coverage=cov.coverage,
                singleton_share=cov.singleton_share,
                mean_score=statistics.fmean(scores) if scores else None,
                survey_count=len(scores),
            )
        )

    chosen = min(points, key=lambda p: abs(p.cluster_count_min2 - target_k))
    return partitions_by_resolution[chosen.resolution], points


def restrict_to_common_support(
    gold: dict[str, dict[str, list[str]]],
    partitions: dict[str, dict[str, int]],
) -> dict[str, dict[str, list[str]]]:
    """Filter every survey's ref lists down to refs placed in a >=2-doc cluster under
    *every* arm in `partitions`. Without this, an arm that refuses to cluster most of the
    corpus (e.g. citation_only on mostly-unkeyed docs) gets scored only on the easy subset
    it did place — which is exactly the citation-only pathology this hardening pass exists
    to close. Surveys/sections left with <2 common refs are dropped (nothing left to score
    a concentration index over)."""
    cluster_sizes = {name: Counter(p.values()) for name, p in partitions.items()}

    def covered_everywhere(doc_id: str) -> bool:
        for name, p in partitions.items():
            cid = p.get(doc_id)
            if cid is None or cluster_sizes[name][cid] < 2:
                return False
        return True

    result: dict[str, dict[str, list[str]]] = {}
    for survey_id, sections in gold.items():
        new_sections = {}
        for section, refs in sections.items():
            filtered = [d for d in refs if covered_everywhere(d)]
            if len(filtered) >= 2:
                new_sections[section] = filtered
        if new_sections:
            result[survey_id] = new_sections
    return result


def sign_test(diffs: list[float]) -> dict[str, Any]:
    """Two-sided exact binomial sign test on the nonzero elements of `diffs` (H0: P(diff >
    0) == 0.5). Ties (exact-zero diffs) are excluded, matching the standard sign-test
    convention."""
    nonzero = [d for d in diffs if d != 0]
    n = len(nonzero)
    if n == 0:
        return {"n": 0, "positive": 0, "negative": 0, "p_value": None}
    positive = sum(1 for d in nonzero if d > 0)
    negative = n - positive
    k = min(positive, negative)
    p_value = min(1.0, sum(math.comb(n, i) for i in range(k + 1)) * 2 / (2**n))
    return {"n": n, "positive": positive, "negative": negative, "p_value": p_value}


def bootstrap_ci(
    diffs: list[float],
    *,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Percentile bootstrap CI on the mean of `diffs` (resample surveys with replacement).
    Small `len(diffs)` (dozens of surveys) is exactly the regime this is for — asymptotic
    normal CIs are unreliable there, and the promotion gate needs to know whether 0 is
    plausible, not just a point estimate."""
    if not diffs:
        return {"mean": None, "ci_low": None, "ci_high": None, "n_boot": n_boot}
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        means.append(statistics.fmean(diffs[rng.randrange(n)] for _ in range(n)))
    means.sort()
    lo = means[max(0, int((alpha / 2) * n_boot))]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot) - 1)]
    return {
        "mean": statistics.fmean(diffs),
        "ci_low": lo,
        "ci_high": hi,
        "n_boot": n_boot,
    }


@dataclass
class PairedComparison:
    against: str
    n_surveys: int
    mean_diff: float | None
    ci_low: float | None
    ci_high: float | None
    sign_test_p: float | None
    excludes_zero: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "against": self.against,
            "n_surveys": self.n_surveys,
            "mean_diff": round(self.mean_diff, 6) if self.mean_diff is not None else None,
            "ci_low": round(self.ci_low, 6) if self.ci_low is not None else None,
            "ci_high": round(self.ci_high, 6) if self.ci_high is not None else None,
            "sign_test_p": round(self.sign_test_p, 6) if self.sign_test_p is not None else None,
            "excludes_zero": self.excludes_zero,
        }


def _survey_mean_score(sections: dict[str, list[str]], partition: dict[str, int]) -> float | None:
    vals = [v for v in (score_refs(refs, partition) for refs in sections.values()) if v is not None]
    return statistics.fmean(vals) if vals else None


def paired_comparison(
    gold: dict[str, dict[str, list[str]]],
    partitions: dict[str, dict[str, int]],
    *,
    hybrid_name: str = "hybrid",
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, PairedComparison]:
    """Per-survey `hybrid - other` concentration-score differences (over common-support
    `gold`), summarized by a sign test and a bootstrap CI on the mean difference — one
    `PairedComparison` per non-hybrid arm present in `partitions`."""
    results: dict[str, PairedComparison] = {}
    hybrid_partition = partitions.get(hybrid_name)
    if hybrid_partition is None:
        return results

    for name, partition in partitions.items():
        if name == hybrid_name:
            continue
        diffs = []
        for _survey_id, sections in sorted(gold.items()):
            h = _survey_mean_score(sections, hybrid_partition)
            o = _survey_mean_score(sections, partition)
            if h is None or o is None:
                continue
            diffs.append(h - o)

        sign = sign_test(diffs)
        boot = bootstrap_ci(diffs, n_boot=n_boot, seed=seed)
        excludes_zero = None
        if boot["ci_low"] is not None:
            excludes_zero = boot["ci_low"] > 0 or boot["ci_high"] < 0
        results[name] = PairedComparison(
            against=name,
            n_surveys=len(diffs),
            mean_diff=boot["mean"],
            ci_low=boot["ci_low"],
            ci_high=boot["ci_high"],
            sign_test_p=sign["p_value"],
            excludes_zero=excludes_zero,
        )
    return results


@dataclass
class AblationArm:
    name: str
    alpha: float
    cluster_count: int
    concentration: ConcentrationReport
    cluster_count_min2: int = 0
    resolution: float = 1.0
    coverage: float | None = None
    singleton_share: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "alpha": self.alpha,
            "cluster_count": self.cluster_count,
            "cluster_count_min2": self.cluster_count_min2,
            "resolution": self.resolution,
            "coverage": round(self.coverage, 6) if self.coverage is not None else None,
            "singleton_share": round(self.singleton_share, 6)
            if self.singleton_share is not None
            else None,
            **self.concentration.to_dict(),
        }


@dataclass
class GateResult:
    verdict: str  # "PROMOTE" | "TIE" | "BLOCK"
    reasons: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.verdict == "PROMOTE"

    def to_dict(self) -> dict[str, Any]:
        return {"gate": self.verdict, "reasons": self.reasons}


def evaluate_promotion_gate(
    arms: dict[str, AblationArm],
    paired: dict[str, PairedComparison],
) -> GateResult:
    """PROMOTE only if hybrid beats BOTH text_only and citation_only with a bootstrap CI
    (on the paired per-survey score difference) that excludes 0 in hybrid's favor. TIE if
    every comparison's CI includes 0 (no arm beats hybrid, but hybrid doesn't clearly win
    either). BLOCK if hybrid is missing, a comparison arm/data is missing, or hybrid loses
    to any arm with a CI excluding 0 in that arm's favor."""
    if "hybrid" not in arms:
        return GateResult(verdict="BLOCK", reasons=["hybrid arm missing — cannot evaluate gate"])

    reasons: list[str] = []
    sub_verdicts: list[str] = []
    for name in ("text_only", "citation_only"):
        if name not in arms:
            reasons.append(f"{name} arm missing — cannot compare, gate blocks")
            sub_verdicts.append("BLOCK")
            continue
        pc = paired.get(name)
        if pc is None or pc.mean_diff is None:
            reasons.append(f"{name}: no paired-comparison data — cannot compare, gate blocks")
            sub_verdicts.append("BLOCK")
            continue
        ci = f"[{pc.ci_low:.4f}, {pc.ci_high:.4f}]" if pc.ci_low is not None else "n/a"
        if pc.excludes_zero and pc.mean_diff > 0:
            reasons.append(
                f"hybrid beats {name}: mean_diff={pc.mean_diff:.4f} CI={ci} "
                f"(n={pc.n_surveys} surveys) excludes 0 — WIN"
            )
            sub_verdicts.append("PROMOTE")
        elif pc.excludes_zero and pc.mean_diff < 0:
            reasons.append(
                f"hybrid LOSES to {name}: mean_diff={pc.mean_diff:.4f} CI={ci} "
                f"(n={pc.n_surveys} surveys) excludes 0 — LOSS"
            )
            sub_verdicts.append("BLOCK")
        else:
            reasons.append(
                f"hybrid vs {name}: mean_diff={pc.mean_diff:.4f} CI={ci} "
                f"(n={pc.n_surveys} surveys) includes 0 — TIE"
            )
            sub_verdicts.append("TIE")

    if "BLOCK" in sub_verdicts:
        verdict = "BLOCK"
    elif all(v == "PROMOTE" for v in sub_verdicts):
        verdict = "PROMOTE"
    else:
        verdict = "TIE"
    return GateResult(verdict=verdict, reasons=reasons)


def run_ablation(
    doc_vectors: dict[str, list[float]],
    adjacency: CitationAdjacency | None,
    gold: dict[str, dict[str, list[str]]],
    *,
    top_k: int = 10,
    seed: int = 42,
    n_random: int = 100,
    target_k: int | None = None,
    resolutions: tuple[float, ...] = DEFAULT_RESOLUTION_GRID,
    variants: tuple[tuple[str, float], ...] = ABLATION_VARIANTS,
    n_boot: int = 2000,
) -> dict[str, Any]:
    """Run the ablation (default hybrid/text-only/citation-only, or `variants=
    ALPHA_SWEEP_VARIANTS` for `--alpha-sweep`) at a shared, granularity-matched `target_k`
    (default: the "hybrid" variant's natural resolution=1.0 >=2-doc cluster count — falls
    back to the first available variant's natural count if there's no arm literally named
    "hybrid", e.g. a custom `variants` list), scored on common-support gold, with a
    paired-comparison promotion gate.

    Skips any variant whose alpha > 0 when `adjacency is None` (no citation store) — the
    output then only has the alpha=0 arm(s), and the gate blocks with a clear reason.
    """
    available = [(name, a) for name, a in variants if not (adjacency is None and a > 0.0)]
    if not available:
        return {
            "schema_version": SCHEMA_VERSION,
            "doc_count": len(doc_vectors),
            "survey_count": len(gold),
            "common_support_survey_count": 0,
            "citation_signal_available": adjacency is not None,
            "target_k": None,
            "arms": {},
            "sweep_curves": {},
            "paired": {},
            "best_alpha": None,
            "gate": GateResult(verdict="BLOCK", reasons=["no variants available to run"]).to_dict(),
        }

    if target_k is None:
        anchor_name, anchor_alpha = next(
            ((n, a) for n, a in available if n == "hybrid"), available[0]
        )
        anchor_edges = build_fused_edges(
            doc_vectors,
            adjacency,
            alpha=(anchor_alpha if adjacency is not None else 0.0),
            top_k=top_k,
        )
        anchor_partition = leiden_communities(
            anchor_edges, all_doc_ids=list(doc_vectors), seed=seed, resolution=1.0
        )
        target_k = max(cluster_count_min2(anchor_partition), 1)

    partitions: dict[str, dict[str, int]] = {}
    sweep_curves: dict[str, list[dict[str, Any]]] = {}
    chosen_resolution: dict[str, float] = {}
    for name, a in available:
        partition, points = sweep_resolution(
            doc_vectors,
            adjacency,
            gold,
            alpha=a,
            target_k=target_k,
            top_k=top_k,
            seed=seed,
            resolutions=resolutions,
        )
        partitions[name] = partition
        sweep_curves[name] = [p.to_dict() for p in points]
        chosen = min(points, key=lambda p: abs(p.cluster_count_min2 - target_k))
        chosen_resolution[name] = chosen.resolution

    common_gold = restrict_to_common_support(gold, partitions)

    arms: dict[str, AblationArm] = {}
    for name, a in available:
        partition = partitions[name]
        report = concentration_report(common_gold, partition, n_random=n_random, seed=seed)
        cov = cluster_coverage(partition, doc_vectors.keys())
        arms[name] = AblationArm(
            name=name,
            alpha=a,
            cluster_count=len(set(partition.values())),
            cluster_count_min2=cluster_count_min2(partition),
            resolution=chosen_resolution[name],
            coverage=cov.coverage,
            singleton_share=cov.singleton_share,
            concentration=report,
        )

    paired = (
        paired_comparison(common_gold, partitions, n_boot=n_boot, seed=seed)
        if "hybrid" in partitions
        else {}
    )
    gate = evaluate_promotion_gate(arms, paired)

    best_alpha = None
    interior = [n for n, a in available if 0.0 < a < 1.0]
    if len(interior) > 1:

        def _z(name: str) -> float:
            z = arms[name].concentration.mean_z_size_matched
            return z if z is not None else float("-inf")

        best_alpha = max(interior, key=_z)

    return {
        "schema_version": SCHEMA_VERSION,
        "doc_count": len(doc_vectors),
        "survey_count": len(gold),
        "common_support_survey_count": len(common_gold),
        "citation_signal_available": adjacency is not None,
        "target_k": target_k,
        "arms": {name: arm.to_dict() for name, arm in arms.items()},
        "sweep_curves": sweep_curves,
        "paired": {name: pc.to_dict() for name, pc in paired.items()},
        "best_alpha": best_alpha,
        "gate": gate.to_dict(),
    }


def render_markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# E1 map-gate report",
        "",
        f"- doc_count: {payload['doc_count']}",
        f"- survey_count: {payload['survey_count']} "
        f"(common-support: {payload.get('common_support_survey_count', 'n/a')})",
        f"- citation_signal_available: {payload['citation_signal_available']}",
        f"- target_k (granularity-matched >=2-doc cluster count): {payload.get('target_k')}",
        "",
        "| Arm | alpha | resolution | clusters | clusters(>=2) | coverage | singleton_share | "
        "surveys scored | mean concentration | z (uniform) | z (size-matched) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, arm in payload["arms"].items():
        mean_score = arm.get("mean_score")
        z_u = arm.get("mean_z_uniform")
        z_sm = arm.get("mean_z_size_matched")
        cov = arm.get("coverage")
        singleton = arm.get("singleton_share")
        lines.append(
            f"| `{name}` | {arm['alpha']:.2f} | {arm.get('resolution', 1.0):.3f} | "
            f"{arm['cluster_count']} | {arm.get('cluster_count_min2', 'n/a')} | "
            f"{'n/a' if cov is None else f'{cov:.3f}'} | "
            f"{'n/a' if singleton is None else f'{singleton:.3f}'} | "
            f"{arm['survey_count']} | "
            f"{'n/a' if mean_score is None else f'{mean_score:.4f}'} | "
            f"{'n/a' if z_u is None else f'{z_u:.3f}'} | "
            f"{'n/a' if z_sm is None else f'{z_sm:.3f}'} |"
        )
    if payload.get("paired"):
        lines += [
            "",
            "## Paired comparison (hybrid minus arm, per common-support survey)",
            "",
            "| vs | n surveys | mean diff | 95% CI | sign-test p | excludes 0 |",
            "| --- | ---: | ---: | --- | ---: | --- |",
        ]
        for name, pc in payload["paired"].items():
            md = pc.get("mean_diff")
            lo, hi = pc.get("ci_low"), pc.get("ci_high")
            p = pc.get("sign_test_p")
            lines.append(
                f"| `{name}` | {pc['n_surveys']} | "
                f"{'n/a' if md is None else f'{md:.4f}'} | "
                f"{'n/a' if lo is None else f'[{lo:.4f}, {hi:.4f}]'} | "
                f"{'n/a' if p is None else f'{p:.4f}'} | {pc.get('excludes_zero')} |"
            )
    if payload.get("best_alpha"):
        lines += ["", f"Best alpha variant (alpha-sweep): `{payload['best_alpha']}`"]
    lines += [
        "",
        "## Promotion gate",
        "",
        f"VERDICT: {payload['gate']['gate']}",
        "",
    ]
    for reason in payload["gate"]["reasons"]:
        lines.append(f"- {reason}")
    lines.append("")
    return "\n".join(lines)
