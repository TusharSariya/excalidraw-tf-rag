"""Ablation harness: hybrid vs text-only vs citation-only clustering, scored by
reference-concentration, with a PROMOTE/BLOCK gate.

Reuses `rag_common.litmap` for every clustering step — `similarity.py` for per-signal kNN
+ RRF fusion, `communities.py` for Leiden partitioning — this module only orchestrates the
three `alpha` settings and folds the results through `concentration.py`.

**Promotion gate (plan/D10):** hybrid is promoted only if it beats BOTH single-signal
ablations on `mean_z_size_matched` (the null-controlled concentration score — see
`concentration.py` docstring for why raw concentration alone is gameable by cluster-size
skew). Ties or missing data never pass silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_common.evals.concentration import ConcentrationReport, concentration_report
from rag_common.litmap.communities import leiden_communities
from rag_common.litmap.similarity import bib_coupling_all, cosine_knn_all, fuse_neighbors
from rag_common.litmap.types import CitationAdjacency

SCHEMA_VERSION = 1

# (label, alpha) — alpha=0 is pure text (cosine kNN only), alpha=1 is pure citation
# (bib-coupling only); both are exact reductions of `fuse_neighbors` (see similarity.py).
ABLATION_VARIANTS: tuple[tuple[str, float], ...] = (
    ("hybrid", 0.5),
    ("text_only", 0.0),
    ("citation_only", 1.0),
)


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
    edges = [(u, v, w) for (u, v), w in edge_scores.items()]
    return leiden_communities(edges, all_doc_ids=doc_ids, seed=seed, resolution=resolution)


@dataclass
class AblationArm:
    name: str
    alpha: float
    cluster_count: int
    concentration: ConcentrationReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "alpha": self.alpha,
            "cluster_count": self.cluster_count,
            **self.concentration.to_dict(),
        }


@dataclass
class GateResult:
    passed: bool
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"gate": "PROMOTE" if self.passed else "BLOCK", "reasons": self.reasons}


def _z(arm: AblationArm | None) -> float | None:
    return arm.concentration.mean_z_size_matched if arm else None


def evaluate_promotion_gate(arms: dict[str, AblationArm]) -> GateResult:
    hybrid = arms.get("hybrid")
    text_only = arms.get("text_only")
    citation_only = arms.get("citation_only")
    reasons: list[str] = []

    hybrid_z = _z(hybrid)
    text_z = _z(text_only)
    citation_z = _z(citation_only)

    if hybrid_z is None:
        reasons.append("hybrid has no size-matched z-score (no gold surveys scored)")
        return GateResult(passed=False, reasons=reasons)

    ok = True
    for label, z in (("text_only", text_z), ("citation_only", citation_z)):
        if z is None:
            reasons.append(f"{label} has no size-matched z-score — cannot compare, gate blocks")
            ok = False
            continue
        beat = hybrid_z > z
        reasons.append(
            f"hybrid z={hybrid_z:.3f} {'>' if beat else '<='} {label} z={z:.3f}: "
            f"{'PASS' if beat else 'FAIL'}"
        )
        ok = ok and beat

    return GateResult(passed=ok, reasons=reasons)


def run_ablation(
    doc_vectors: dict[str, list[float]],
    adjacency: CitationAdjacency | None,
    gold: dict[str, dict[str, list[str]]],
    *,
    top_k: int = 10,
    seed: int = 42,
    resolution: float = 1.0,
    n_random: int = 100,
) -> dict[str, Any]:
    """Run the hybrid/text-only/citation-only ablation and score each against `gold`
    (survey_doc_id -> {section -> [ref_doc_id,...]}, see `survey_gold.to_gold_dict`).

    Skips `citation_only`/`hybrid` when `adjacency is None` (no citation store) — the
    output then only has `text_only`, and the gate blocks with a clear reason rather than
    comparing against a signal that doesn't exist.
    """
    arms: dict[str, AblationArm] = {}
    for name, alpha in ABLATION_VARIANTS:
        if adjacency is None and alpha > 0.0:
            continue
        partition = build_partition(
            doc_vectors, adjacency, alpha=alpha, top_k=top_k, seed=seed, resolution=resolution
        )
        report = concentration_report(gold, partition, n_random=n_random, seed=seed)
        arms[name] = AblationArm(
            name=name,
            alpha=alpha,
            cluster_count=len(set(partition.values())),
            concentration=report,
        )

    gate = evaluate_promotion_gate(arms)
    return {
        "schema_version": SCHEMA_VERSION,
        "doc_count": len(doc_vectors),
        "survey_count": len(gold),
        "citation_signal_available": adjacency is not None,
        "arms": {name: arm.to_dict() for name, arm in arms.items()},
        "gate": gate.to_dict(),
    }


def render_markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# E1 map-gate report",
        "",
        f"- doc_count: {payload['doc_count']}",
        f"- survey_count: {payload['survey_count']}",
        f"- citation_signal_available: {payload['citation_signal_available']}",
        "",
        "| Arm | alpha | clusters | surveys scored | mean concentration | "
        "z (uniform) | z (size-matched) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, arm in payload["arms"].items():
        mean_score = arm.get("mean_score")
        z_u = arm.get("mean_z_uniform")
        z_sm = arm.get("mean_z_size_matched")
        lines.append(
            f"| `{name}` | {arm['alpha']:.2f} | {arm['cluster_count']} | "
            f"{arm['survey_count']} | "
            f"{'n/a' if mean_score is None else f'{mean_score:.4f}'} | "
            f"{'n/a' if z_u is None else f'{z_u:.3f}'} | "
            f"{'n/a' if z_sm is None else f'{z_sm:.3f}'} |"
        )
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
