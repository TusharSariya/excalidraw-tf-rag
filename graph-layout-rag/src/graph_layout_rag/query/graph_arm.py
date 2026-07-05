"""HippoRAG-2-style graph retrieval arm (LLM-free at query time).

A relational/multi-hop retrieval arm that complements the plain hybrid search. The shape
follows HippoRAG-2 (2026): retrieval *is* a graph algorithm — no LLM call at query time.

Pipeline:
  1. **Seed selection** — run the normal hybrid retrieval, map the top chunks to canonical
     docs and keep each doc's best hybrid score. These scores become the personalized
     PageRank restart distribution (∝ normalized seed scores), so the walk starts biased
     toward the query-relevant docs rather than a uniform teleport.
  2. **Graph** — an undirected weighted adjacency over docs, the UNION of two edge sources:
       * citation edges — the in-corpus projection of each tool's ``CitationGraph``
         (``undirected`` adjacency mapped oa-id -> corpus doc_id);
       * embedding-kNN edges — cosine top-k over the litmap doc vectors (mean-pooled from
         the champion LanceDB chunk index). The doc vectors are reused from litmap's cache;
         the derived kNN adjacency is cached to ``<index>/graph_arm_knn.json`` with a
         version key (schema + embed_profile + index_fingerprint + k) so a re-index or a
         different k rebuilds it.
  3. **Personalized PageRank** — a weighted random-walk-with-restart seeded by the hybrid
     scores (``seeded_pagerank``, same algorithm family as ``citation_rank`` PPR: dangling
     mass and restart both redistribute over the teleport distribution).
  4. **Blend** — ``alpha * PPR + (1 - alpha) * hybrid`` per doc (both max-normalized),
     ``alpha`` default 0.5 (env ``RAG_GRAPH_ARM_ALPHA``). Returns the SAME doc-level result
     shape as ``query.search.search`` (via ``format_results``) so all downstream rendering
     works unchanged. Chunks for a returned doc come from the seed pool when present, else
     are fetched directly from the champion index (no new embedding compute).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

SCHEMA_VERSION = 1

DEFAULT_ALPHA = 0.5  # blend weight on the PPR signal; (1 - alpha) on hybrid
DEFAULT_RESTART = 0.5  # RWR restart probability — high, to stay near the seeds
DEFAULT_KNN_K = 15  # embedding-kNN edges kept per doc
DEFAULT_SEED_TOP = 30  # seed chunks pulled from the hybrid arm
DEFAULT_CITE_WEIGHT = 1.0
DEFAULT_KNN_WEIGHT = 1.0

_KNN_CACHE_NAME = "graph_arm_knn.json"


# --------------------------------------------------------------------------- #
# Env-tunable knobs
# --------------------------------------------------------------------------- #
def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def alpha_from_env() -> float:
    """Blend weight on the PPR signal, clamped to [0, 1]."""
    return min(1.0, max(0.0, _env_float("RAG_GRAPH_ARM_ALPHA", DEFAULT_ALPHA)))


def restart_from_env() -> float:
    return min(0.99, max(0.01, _env_float("RAG_GRAPH_ARM_RESTART", DEFAULT_RESTART)))


# --------------------------------------------------------------------------- #
# Seed scoring
# --------------------------------------------------------------------------- #
def _row_score(row: dict[str, Any]) -> float:
    for key in ("fusion_score", "score"):
        val = row.get(key)
        if val is not None:
            return float(val)
    return 0.0


def seed_doc_scores(
    candidates: list[dict[str, Any]],
    canon: Callable[[dict[str, Any]], str],
) -> dict[str, float]:
    """Best hybrid score per canonical doc from the seed chunk candidates.

    ``canon(row)`` maps a chunk row to its canonical doc id (e.g.
    ``identities.canonical_doc_id``). A doc's score is the max over its chunks.
    """
    scores: dict[str, float] = {}
    for row in candidates:
        doc = canon(row)
        s = _row_score(row)
        if s > scores.get(doc, float("-inf")):
            scores[doc] = s
    return scores


def normalize_restart(seed_scores: dict[str, float]) -> dict[str, float]:
    """Restart distribution ∝ (positive) seed scores, summing to 1.

    Falls back to a uniform distribution over the seeds when every score is
    non-positive (so a degenerate all-zero seed set still teleports somewhere).
    """
    positives = {d: v for d, v in seed_scores.items() if v > 0}
    total = sum(positives.values())
    if total <= 0.0:
        n = len(seed_scores)
        return {d: 1.0 / n for d in seed_scores} if n else {}
    return {d: positives.get(d, 0.0) / total for d in seed_scores}


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
def knn_edges(
    vectors: dict[str, list[float]],
    *,
    k: int = DEFAULT_KNN_K,
    canon: Callable[[str], str] | None = None,
) -> dict[str, dict[str, float]]:
    """Symmetric cosine-kNN adjacency over the doc vectors (weight = cosine, clamped ≥ 0).

    Reuses ``rag_common.litmap.similarity.cosine_knn_all`` (one matmul). ``canon`` collapses
    alias doc ids so both endpoints live in canonical-doc space.
    """
    from rag_common.litmap.similarity import cosine_knn_all

    c = canon or (lambda d: d)
    neighbors = cosine_knn_all(vectors, top_k=k)
    edges: dict[str, dict[str, float]] = {}
    for doc, nbrs in neighbors.items():
        cu = c(doc)
        for n in nbrs:
            cv = c(n.doc_id)
            if cu == cv:
                continue
            w = max(0.0, float(n.score))
            # keep the strongest cosine for a canonical pair (aliases can dup)
            if w > edges.setdefault(cu, {}).get(cv, 0.0):
                edges[cu][cv] = w
                edges.setdefault(cv, {})[cu] = w
    return edges


def citation_doc_edges(
    graph: Any,
    *,
    canon: Callable[[str], str] | None = None,
) -> dict[str, set[str]]:
    """In-corpus citation adjacency projected into canonical-doc space.

    For every corpus oa node, connect it (undirected) to every citation neighbor that also
    maps to a corpus doc. Non-corpus references/citers are dropped (no doc to rank), which
    is exactly the ``in_corpus`` projection the multi-hop eval pairs seeds on.
    """
    if graph is None:
        return {}
    c = canon or (lambda d: d)
    oa_to_doc = getattr(graph, "oa_to_doc", {})
    undirected = getattr(graph, "undirected", {})
    edges: dict[str, set[str]] = {}
    for oa, doc in oa_to_doc.items():
        cd = c(doc)
        for nb in undirected.get(oa, ()):  # type: ignore[union-attr]
            nd = oa_to_doc.get(nb)
            if not nd:
                continue
            cnd = c(nd)
            if cnd == cd:
                continue
            edges.setdefault(cd, set()).add(cnd)
            edges.setdefault(cnd, set()).add(cd)
    return edges


def union_graph(
    knn: dict[str, dict[str, float]],
    cite: dict[str, set[str]],
    *,
    knn_weight: float = DEFAULT_KNN_WEIGHT,
    cite_weight: float = DEFAULT_CITE_WEIGHT,
) -> dict[str, dict[str, float]]:
    """Symmetric weighted union of the kNN and citation adjacencies.

    A pair connected by both signals sums its weights (kNN cosine * ``knn_weight`` plus a
    flat ``cite_weight``), so co-cited AND embedding-similar docs bind more tightly.
    """
    weights: dict[tuple[str, str], float] = {}

    def _key(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    for u, nbrs in knn.items():
        for v, w in nbrs.items():
            weights[_key(u, v)] = max(weights.get(_key(u, v), 0.0), knn_weight * max(0.0, w))
    # Citation edges are symmetric (both u->v and v->u appear); add the flat
    # cite_weight once per unordered pair so it is not double-counted.
    cite_pairs = {_key(u, v) for u, nbrs in cite.items() for v in nbrs}
    for pair in cite_pairs:
        weights[pair] = weights.get(pair, 0.0) + cite_weight

    adj: dict[str, dict[str, float]] = {}
    for (a, b), w in weights.items():
        adj.setdefault(a, {})[b] = w
        adj.setdefault(b, {})[a] = w
    return adj


# --------------------------------------------------------------------------- #
# Weighted random walk with restart (personalized PageRank)
# --------------------------------------------------------------------------- #
def seeded_pagerank(
    adj: dict[str, dict[str, float]],
    restart_weights: dict[str, float],
    *,
    restart: float = DEFAULT_RESTART,
    iters: int = 60,
    tol: float = 1e-9,
) -> dict[str, float]:
    """Weighted RWR over ``adj`` teleporting to ``restart_weights``.

    Same shape as ``citation_rank.personalized_pagerank`` but (a) edges carry weights and
    (b) the teleport distribution is the caller-supplied (hybrid-derived) one rather than
    uniform-over-seeds. Dangling mass and the restart term both redistribute over the
    teleport distribution. ``restart_weights`` need not be normalized — it is normalized
    here. Returns the stationary score per node (seeds included).
    """
    tele_total = sum(w for w in restart_weights.values() if w > 0)
    if tele_total > 0:
        tele = {n: (w / tele_total if w > 0 else 0.0) for n, w in restart_weights.items()}
    elif restart_weights:
        u = 1.0 / len(restart_weights)
        tele = {n: u for n in restart_weights}
    else:
        return {}

    nodes = set(adj) | set(tele)
    out = {n: sum(adj.get(n, {}).values()) for n in nodes}
    rank = {n: tele.get(n, 0.0) for n in nodes}

    for _ in range(iters):
        nxt = {n: 0.0 for n in nodes}
        dangling = 0.0
        for n, score in rank.items():
            o = out.get(n, 0.0)
            if o <= 0.0:
                dangling += score
                continue
            for m, w in adj[n].items():
                nxt[m] += score * (w / o)
        delta = 0.0
        for n in nodes:
            walked = (1.0 - restart) * (nxt[n] + dangling * tele.get(n, 0.0))
            val = walked + restart * tele.get(n, 0.0)
            delta += abs(val - rank[n])
            nxt[n] = val
        rank = nxt
        if delta < tol:
            break
    return rank


def _max_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    peak = max(scores.values())
    if peak <= 0.0:
        return {k: 0.0 for k in scores}
    return {k: v / peak for k, v in scores.items()}


def blend_scores(
    ppr: dict[str, float],
    hybrid: dict[str, float],
    *,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, float]:
    """``alpha * PPR + (1 - alpha) * hybrid`` per doc, each max-normalized to [0, 1].

    ``alpha=0`` reduces exactly to the (normalized) hybrid ranking; ``alpha=1`` to the pure
    PPR ranking. Docs present in only one signal contribute 0 from the other.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha!r}")
    pn = _max_normalize(ppr)
    hn = _max_normalize(hybrid)
    docs = set(pn) | set(hn)
    return {d: alpha * pn.get(d, 0.0) + (1.0 - alpha) * hn.get(d, 0.0) for d in docs}


# --------------------------------------------------------------------------- #
# kNN adjacency cache (reuses litmap doc vectors)
# --------------------------------------------------------------------------- #
def load_or_build_knn(
    profile: str | None = None,
    *,
    k: int = DEFAULT_KNN_K,
    rebuild: bool = False,
) -> tuple[dict[str, dict[str, float]], Any]:
    """Load the cached graph-arm kNN adjacency, rebuilding on a version-key miss.

    The version key is (schema, embed_profile, index_fingerprint, k): a re-index (new
    fingerprint) or a different ``k`` invalidates the cache. Doc vectors themselves are
    reused from litmap's own cache via ``load_or_build_doc_vectors`` — no new embedding.
    Returns ``(edges, ProfileIndexPaths)``. Edges are keyed by *raw* doc id here (the
    caller canonicalizes when building the union graph).
    """
    from graph_layout_rag.query.litmap_support import (
        _index_fingerprint,
        load_or_build_doc_vectors,
    )

    vectors, paths = load_or_build_doc_vectors(profile)
    fingerprint = _index_fingerprint(paths.lance_dir)
    cache_path = paths.root / _KNN_CACHE_NAME

    if not rebuild and cache_path.exists():
        cached = _read_knn_cache(cache_path, profile=paths.profile, fingerprint=fingerprint, k=k)
        if cached is not None:
            return cached, paths

    edges = knn_edges(vectors, k=k)
    _write_knn_cache(cache_path, edges, profile=paths.profile, fingerprint=fingerprint, k=k)
    return edges, paths


def _read_knn_cache(
    path: Any,
    *,
    profile: str,
    fingerprint: str,
    k: int,
) -> dict[str, dict[str, float]] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if (
        data.get("schema_version") != SCHEMA_VERSION
        or data.get("embed_profile") != profile
        or data.get("index_fingerprint") != fingerprint
        or data.get("k") != k
    ):
        return None
    return {u: {v: float(w) for v, w in nbrs.items()} for u, nbrs in data.get("edges", {}).items()}


def _write_knn_cache(
    path: Any,
    edges: dict[str, dict[str, float]],
    *,
    profile: str,
    fingerprint: str,
    k: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "embed_profile": profile,
        "index_fingerprint": fingerprint,
        "k": k,
        "edges": edges,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Result-row assembly (shared, pure — testable without an index)
# --------------------------------------------------------------------------- #
def assemble_rows(
    ranked_docs: list[str],
    blended: dict[str, float],
    pool_by_doc: dict[str, list[dict[str, Any]]],
    fetch_fn: Callable[[str], list[dict[str, Any]]],
    *,
    top: int,
    max_per_doc: int,
) -> list[dict[str, Any]]:
    """Build chunk rows for the top-blended docs, ordered by blend score.

    Reuses the seed pool's chunk rows for docs already retrieved; falls back to
    ``fetch_fn(doc)`` (a direct index lookup) for graph-surfaced docs not in the pool. Each
    emitted row carries the doc's blend score as ``fusion_score`` (and drops any
    ``rerank_score``) so ``format_results`` orders by the blend and reports it as the score.
    """
    rows: list[dict[str, Any]] = []
    used = 0
    for doc in ranked_docs:
        if used >= top:
            break
        doc_rows = pool_by_doc.get(doc) or fetch_fn(doc)
        if not doc_rows:
            continue
        score = round(blended.get(doc, 0.0), 6)
        for r in doc_rows[:max_per_doc]:
            r = dict(r)
            r.pop("rerank_score", None)
            r["fusion_score"] = score
            rows.append(r)
        used += 1
    return rows


def _fetch_doc_chunks(
    ctx: Any,
    canonical_doc: str,
    identities: Any,
    limit: int,
) -> list[dict[str, Any]]:
    """Pull up to ``limit`` chunk rows for a doc straight from the champion index.

    No embedding — a filter-only LanceDB scan over the doc's raw id(s). Degrades to an empty
    list on any error (the doc is simply skipped in the results).
    """
    try:
        raw_ids = list(identities.component_doc_ids(canonical_doc))
    except Exception:
        raw_ids = []
    if not raw_ids:
        raw_ids = [canonical_doc]
    escaped = ", ".join("'" + rid.replace("'", "''") + "'" for rid in raw_ids)
    try:
        rows = ctx.table.search().where(f"doc_id IN ({escaped})").limit(limit).to_list()
    except Exception:
        return []
    for r in rows:
        r.pop("vector", None)
    return rows


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def graph_search(
    query: str,
    *,
    top: int = 8,
    embed_profile: str | None = None,
    alpha: float | None = None,
    restart: float | None = None,
    k: int = DEFAULT_KNN_K,
    seed_top: int = DEFAULT_SEED_TOP,
    max_per_doc: int = 2,
    filters: Any | None = None,
    hybrid: bool = True,
    knn_weight: float = DEFAULT_KNN_WEIGHT,
    cite_weight: float = DEFAULT_CITE_WEIGHT,
) -> list[dict[str, Any]]:
    """Graph retrieval arm. Returns the same doc-level shape as ``query.search.search``."""
    from graph_layout_rag.query.citation_rank import load_graph_cached
    from graph_layout_rag.query.identity import canonical_identity_map
    from graph_layout_rag.query.retrieve import (
        RetrieveFilters,
        resolve_retrieve_context,
        retrieve_candidates,
    )
    from graph_layout_rag.query.search import format_results

    alpha = alpha_from_env() if alpha is None else alpha
    restart = restart_from_env() if restart is None else restart
    filters = filters or RetrieveFilters()
    identities = canonical_identity_map()

    ctx = resolve_retrieve_context(embed_profile=embed_profile)
    seed_candidates = retrieve_candidates(
        query,
        top=max(seed_top, top * 3),
        embed_profile=embed_profile,
        hybrid=hybrid,
        filters=filters,
        context=ctx,
    )
    hybrid_scores = seed_doc_scores(seed_candidates, identities.canonical_doc_id)
    if not hybrid_scores:
        return []

    restart_weights = normalize_restart(hybrid_scores)

    def _canon(doc_id: str) -> str:
        return identities.canonicalize_doc_id(doc_id)

    knn, _paths = load_or_build_knn(embed_profile, k=k)
    knn_c = {_canon(u): {_canon(v): w for v, w in nbrs.items()} for u, nbrs in knn.items()}
    graph = load_graph_cached()
    cite = citation_doc_edges(graph, canon=_canon)
    adj = union_graph(knn_c, cite, knn_weight=knn_weight, cite_weight=cite_weight)

    ppr = seeded_pagerank(adj, restart_weights, restart=restart)
    blended = blend_scores(ppr, hybrid_scores, alpha=alpha)
    ranked_docs = sorted(blended, key=lambda d: (blended[d], d), reverse=True)

    pool_by_doc: dict[str, list[dict[str, Any]]] = {}
    for row in seed_candidates:
        pool_by_doc.setdefault(identities.canonical_doc_id(row), []).append(row)

    rows = assemble_rows(
        ranked_docs,
        blended,
        pool_by_doc,
        lambda doc: _fetch_doc_chunks(ctx, doc, identities, max_per_doc),
        top=top,
        max_per_doc=max_per_doc,
    )
    return format_results(rows, top=top, max_per_doc=max_per_doc)
