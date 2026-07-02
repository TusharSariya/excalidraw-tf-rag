"""Per-signal top-k neighbor lists, fused by rank — not raw score.

Two signals feed the fusion: bibliographic coupling (share outgoing citations — reuses the
injected `CitationAdjacency`, same IDF-weighted formula as each tool's
`query/citation_rank.bibliographic_coupling`) and cosine kNN over doc vectors (plain numpy
matmul, fine below ~10k docs). The two are not commensurate — an IDF-weighted shared-count
sum and a bounded cosine similarity live on different scales — so they're fused with
reciprocal-rank fusion (RRF) instead of a weighted score blend (D10 hardening item, codex
review). `alpha` interpolates between the two signals' RRF weight: `alpha=0` is pure text
(cosine-only), `alpha=1` is pure citation (coupling-only); both reductions are exact and
unit-tested.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from rag_common.litmap.types import CitationAdjacency, Neighbor

DEFAULT_RRF_K = 60


def bib_coupling_neighbors(
    seed_doc_id: str,
    candidate_doc_ids: Iterable[str],
    adj: CitationAdjacency,
    *,
    top_k: int = 20,
) -> list[Neighbor]:
    """IDF-weighted bibliographic coupling: candidates ranked by shared outgoing
    references with the seed. Empty when the seed or every candidate is un-keyed, or the
    seed has no references (isolated in the reference direction)."""
    seed_oa = adj.doc_to_oa.get(seed_doc_id)
    if seed_oa is None:
        return []
    seed_refs = adj.out_adj.get(seed_oa, set())
    if not seed_refs:
        return []
    scored: list[tuple[str, float]] = []
    for cand in candidate_doc_ids:
        if cand == seed_doc_id:
            continue
        cand_oa = adj.doc_to_oa.get(cand)
        if cand_oa is None:
            continue
        shared = seed_refs & adj.out_adj.get(cand_oa, set())
        if not shared:
            continue
        score = sum(adj.idf(r) for r in shared)
        scored.append((cand, score))
    scored.sort(key=lambda t: t[1], reverse=True)
    return [
        Neighbor(doc_id=doc_id, rank=i + 1, score=score)
        for i, (doc_id, score) in enumerate(scored[:top_k])
    ]


def co_citation_neighbors(
    seed_doc_id: str,
    candidate_doc_ids: Iterable[str],
    adj: CitationAdjacency,
    *,
    top_k: int = 20,
) -> list[Neighbor]:
    """IDF-weighted co-citation: candidates ranked by shared citers with the seed."""
    seed_oa = adj.doc_to_oa.get(seed_doc_id)
    if seed_oa is None:
        return []
    seed_citers = adj.in_adj.get(seed_oa, set())
    if not seed_citers:
        return []
    scored: list[tuple[str, float]] = []
    for cand in candidate_doc_ids:
        if cand == seed_doc_id:
            continue
        cand_oa = adj.doc_to_oa.get(cand)
        if cand_oa is None:
            continue
        shared = seed_citers & adj.in_adj.get(cand_oa, set())
        if not shared:
            continue
        score = sum(adj.idf(c) for c in shared)
        scored.append((cand, score))
    scored.sort(key=lambda t: t[1], reverse=True)
    return [
        Neighbor(doc_id=doc_id, rank=i + 1, score=score)
        for i, (doc_id, score) in enumerate(scored[:top_k])
    ]


def cosine_knn(
    seed_doc_id: str,
    doc_vectors: dict[str, list[float]],
    *,
    top_k: int = 20,
) -> list[Neighbor]:
    """Cosine-nearest docs to the seed among `doc_vectors` (assumed L2-normalized, so
    cosine similarity is a dot product). Plain numpy matmul — fine at <10k docs."""
    seed = doc_vectors.get(seed_doc_id)
    if seed is None:
        return []
    import numpy as np

    cand_ids = [d for d in doc_vectors if d != seed_doc_id]
    if not cand_ids:
        return []
    mat = np.array([doc_vectors[d] for d in cand_ids], dtype=np.float32)
    seed_vec = np.array(seed, dtype=np.float32)
    scores = mat @ seed_vec
    order = np.argsort(-scores)[:top_k]
    return [
        Neighbor(doc_id=cand_ids[i], rank=rank + 1, score=float(scores[i]))
        for rank, i in enumerate(order)
    ]


def cosine_knn_all(
    doc_vectors: dict[str, list[float]],
    *,
    top_k: int = 20,
) -> dict[str, list[Neighbor]]:
    """Cosine top-k for every doc against every other doc, via one matmul instead of one
    call per seed — `map clusters` needs this for every doc in the corpus, and doing that
    as `len(doc_vectors)` separate `cosine_knn` calls pays the numpy call overhead
    `len(doc_vectors))` times over. Still O(n^2) memory/compute, fine below ~10k docs."""
    doc_ids = list(doc_vectors)
    if len(doc_ids) < 2:
        return {d: [] for d in doc_ids}
    import numpy as np

    mat = np.array([doc_vectors[d] for d in doc_ids], dtype=np.float32)
    sims = mat @ mat.T
    n = len(doc_ids)
    idx_range = np.arange(n)
    sims[idx_range, idx_range] = -np.inf
    k = min(top_k, n - 1)
    result: dict[str, list[Neighbor]] = {}
    for i, doc_id in enumerate(doc_ids):
        row = sims[i]
        top_idx = np.argpartition(-row, k - 1)[:k]
        top_idx = top_idx[np.argsort(-row[top_idx])]
        result[doc_id] = [
            Neighbor(doc_id=doc_ids[j], rank=rank + 1, score=float(row[j]))
            for rank, j in enumerate(top_idx)
        ]
    return result


def bib_coupling_all(
    doc_ids: Iterable[str],
    adj: CitationAdjacency,
    *,
    top_k: int = 20,
) -> dict[str, list[Neighbor]]:
    """`bib_coupling_neighbors` for every doc, restricted to docs the citation graph can
    place at all (`doc_to_oa`) — un-keyed docs get an empty list rather than paying an
    O(n) scan that can never find a shared reference."""
    keyed = [d for d in doc_ids if d in adj.doc_to_oa]
    return {
        doc_id: bib_coupling_neighbors(doc_id, keyed, adj, top_k=top_k)
        if doc_id in adj.doc_to_oa
        else []
        for doc_id in doc_ids
    }


def rrf_fuse(
    signal_neighbors: dict[str, list[Neighbor]],
    signal_weights: dict[str, float],
    *,
    rrf_k: int = DEFAULT_RRF_K,
    top_k: int = 20,
) -> list[Neighbor]:
    """Fuse multiple ranked neighbor lists by weighted reciprocal-rank fusion.

    `score(doc) = sum_signal weight_signal / (rrf_k + rank_signal(doc))`. A signal with
    weight 0 contributes nothing, so setting weights to `{"text": 1, "citation": 0}` (or
    vice versa) is an exact reduction to that single signal's own ranking — this is what
    `fuse_neighbors`'s `alpha=0`/`alpha=1` cases rely on.
    """
    scores: dict[str, float] = defaultdict(float)
    for signal, neighbors in signal_neighbors.items():
        weight = signal_weights.get(signal, 0.0)
        if weight == 0.0:
            continue
        for n in neighbors:
            scores[n.doc_id] += weight / (rrf_k + n.rank)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [
        Neighbor(doc_id=doc_id, rank=i + 1, score=score) for i, (doc_id, score) in enumerate(ranked)
    ]


def fuse_neighbors(
    text_neighbors: list[Neighbor],
    citation_neighbors: list[Neighbor],
    *,
    alpha: float = 0.5,
    rrf_k: int = DEFAULT_RRF_K,
    top_k: int = 20,
) -> list[Neighbor]:
    """RRF-fuse a text (cosine kNN) and a citation (bib-coupling) neighbor list.

    `alpha` is the citation signal's weight (`1 - alpha` goes to text): `alpha=0` reduces
    exactly to the text ranking, `alpha=1` reduces exactly to the citation ranking.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha!r}")
    return rrf_fuse(
        {"text": text_neighbors, "citation": citation_neighbors},
        {"text": 1.0 - alpha, "citation": alpha},
        rrf_k=rrf_k,
        top_k=top_k,
    )
