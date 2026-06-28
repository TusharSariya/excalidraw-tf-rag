from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from graph_layout_rag.query.citation_rank import CitationGraph

# Citation/recency ranking prior (T7, default OFF). The raw per-doc prior blends
# a log citation-count term with a small linear recency term, then is min-max
# normalized across the live candidate set so it lands on the RRF score scale
# (fusion scores are tiny, ~0.01-0.05) and cannot dominate. Recency uses a
# documented monotonic ramp: max(0, year - RECENCY_YEAR0), scaled down so a
# decade of newness is worth ~1 unit of log-citation before normalization. A
# missing year contributes 0 (neutral; never penalized).
RECENCY_YEAR0 = 2000
RECENCY_PRIOR_SCALE = 0.1

# RRF rank constant. k=20 sharpens top-rank weighting vs the classic 60; the
# de-biased bake-off (neutral LLM-judged qrels) measured it a consistent win
# over k=60 on both tracks (catalog 0.768 vs 0.765, pdf 0.715 vs 0.696).
RRF_K = 20

# Default fusion weights. Weight sweep (2026-06-22 self-improve campaign,
# sparse_weight in {0.3,0.5,0.7,1.0,1.5,2.0} at dense_weight=1.0, rrf_k=20,
# pool=80) found nDCG@10 rising monotonically with sparse_weight across the
# whole grid (no plateau) -- BM25 dominates this corpus much more than the
# 2026-06 bake-off implied (bm25-only: catalog 0.878, pdf 0.878; dense-only:
# catalog 0.615, pdf 0.621). sparse_weight=2.0 clears the calibrated
# promotion gate vs the prior equal-weight default (+0.054 catalog / +0.063
# pdf, no failure increase) but is still BELOW bm25-only on pdf-deep-read
# (0.875 vs 0.878) and only marginally above it on catalog (0.880 vs 0.878).
# Read this as: hybrid fusion adds no real value over BM25 alone on this
# corpus today; sparse_weight=2.0 mainly claws back the equal-weight
# baseline's loss to BM25, it does not represent a verified hybrid win.
# Recorded honestly rather than oversold -- see docs/quality-campaign-2026-06-22.md.
DENSE_WEIGHT = 1.0
SPARSE_WEIGHT = 2.0


def merge_rankings(
    *result_lists: list[dict[str, Any]],
    top: int = 30,
    rrf_k: int = RRF_K,
) -> list[dict[str, Any]]:
    """Merge one or more ranked result lists with reciprocal rank fusion."""
    scores: dict[str, float] = {}
    payloads: dict[str, dict[str, Any]] = {}

    for result_list in result_lists:
        for rank, row in enumerate(result_list, start=1):
            chunk_id = row["id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            base = payloads.get(chunk_id, {})
            base.update(row)
            payloads[chunk_id] = base

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top]
    merged: list[dict[str, Any]] = []
    for chunk_id, score in ranked:
        row = payloads[chunk_id]
        row["id"] = chunk_id
        row["fusion_score"] = round(score, 6)
        merged.append(row)
    return merged


def _raw_doc_prior(graph: CitationGraph, payload: dict[str, Any]) -> float:
    """Raw (un-normalized) doc-level prior for one fused chunk row.

    Citation term: ``citation_prior`` (``log1p`` of global cited-by) for the doc's
    OA node; 0 when the doc_id is absent from ``doc_to_oa`` (neutral, never raises).
    Recency term: a monotonic ramp ``max(0, year - RECENCY_YEAR0)`` scaled by
    ``RECENCY_PRIOR_SCALE``; 0 when ``year`` is missing (neutral, never penalizes).
    The prior is doc-level but applied to the chunk-level fused score by the caller.
    """
    from graph_layout_rag.query.citation_rank import citation_prior

    doc_id = payload.get("doc_id") or ""
    oa = graph.doc_to_oa.get(doc_id)
    citation = citation_prior(graph, oa) if oa else 0.0

    year = payload.get("year")
    recency = RECENCY_PRIOR_SCALE * max(0, int(year) - RECENCY_YEAR0) if year else 0.0
    return citation + recency


def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    *,
    top: int = 30,
    rrf_k: int = RRF_K,
    dense_weight: float = DENSE_WEIGHT,
    sparse_weight: float = SPARSE_WEIGHT,
    citation_prior_weight: float = 0.0,
    citation_graph: CitationGraph | None = None,
) -> list[dict[str, Any]]:
    """Merge dense (vector) and sparse (BM25) rankings with reciprocal rank fusion.

    Rows are keyed by their ``id`` (``"{doc_id}:{chunk_index}"``). Later lists
    overwrite field collisions; dense payloads typically win when passed last.

    ``citation_prior_weight`` (default 0.0) optionally blends a doc-level
    citation/recency prior into the fused score. At the default 0.0 — or when no
    ``citation_graph`` is supplied — the prior block is skipped entirely and the
    output is byte-identical (same order and same ``fusion_score`` values) to the
    no-prior path. When > 0.0 the raw prior (see ``_raw_doc_prior``) is min-max
    normalized across the live candidate set into [0, 1] and blended as
    ``fused_score += citation_prior_weight * normalized_prior`` BEFORE ranking, so
    a small weight cannot let the prior dominate the tiny (~0.01-0.05) RRF scores.
    """
    scores: dict[str, float] = {}
    payloads: dict[str, dict[str, Any]] = {}

    for rank, row in enumerate(sparse_results, start=1):
        chunk_id = row["id"]
        scores[chunk_id] = scores.get(chunk_id, 0.0) + sparse_weight / (rrf_k + rank)
        payloads[chunk_id] = {**row, "sparse_rank": rank, "sparse_score": row.get("score")}

    for rank, row in enumerate(dense_results, start=1):
        chunk_id = row["id"]
        scores[chunk_id] = scores.get(chunk_id, 0.0) + dense_weight / (rrf_k + rank)
        base = payloads.get(chunk_id, {})
        base.update(row)
        base["dense_rank"] = rank
        base["dense_score"] = row.get("score")
        payloads[chunk_id] = base

    # T7 prior blend. Guarded so the default (weight 0.0 / no graph) is a true
    # no-op: scores and ordering stay byte-identical to the pre-T7 code path.
    if citation_prior_weight > 0.0 and citation_graph is not None:
        raw_priors = {
            chunk_id: _raw_doc_prior(citation_graph, payload)
            for chunk_id, payload in payloads.items()
        }
        if raw_priors:
            lo = min(raw_priors.values())
            hi = max(raw_priors.values())
            span = hi - lo
            if span > 0:  # all-equal priors normalize to a uniform shift = neutral
                for chunk_id, raw in raw_priors.items():
                    scores[chunk_id] += citation_prior_weight * ((raw - lo) / span)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top]
    merged: list[dict[str, Any]] = []
    for chunk_id, score in ranked:
        row = payloads[chunk_id]
        row["id"] = chunk_id
        row["fusion_score"] = round(score, 6)
        merged.append(row)
    return merged
