"""Connected-Papers-style local neighborhood around a seed document.

Unlike `similarity.fuse_neighbors` (rank-fused, cross-modal text+citation), this is a
single-modality view: similarity is a weighted blend of *raw* bibliographic-coupling and
co-citation scores, both IDF-weighted shared-count sums from the same citation graph — they
share units, so a raw blend (not RRF) is appropriate here, matching the plan's "similarity =
weighted co-citation + bibliographic coupling" spec for this view specifically.
"""

from __future__ import annotations

from collections import defaultdict

from rag_common.litmap.similarity import bib_coupling_neighbors, co_citation_neighbors
from rag_common.litmap.types import CitationAdjacency

SCHEMA_VERSION = 1


def neighborhood_graph(
    seed_doc_id: str,
    candidate_doc_ids: list[str],
    adj: CitationAdjacency,
    *,
    top_n: int = 20,
    coupling_weight: float = 0.5,
    cocitation_weight: float = 0.5,
    threshold: float = 0.0,
    signal_top_k: int = 200,
) -> dict:
    """Top-`top_n` docs by weighted-blended coupling+co-citation similarity to the seed,
    above `threshold`. Returns a JSON graph: `{"nodes": [...], "edges": [...]}`, star-shaped
    around the seed (seed -> neighbor, weighted by the combined score)."""
    coupling = bib_coupling_neighbors(seed_doc_id, candidate_doc_ids, adj, top_k=signal_top_k)
    cocitation = co_citation_neighbors(seed_doc_id, candidate_doc_ids, adj, top_k=signal_top_k)

    combined: dict[str, float] = defaultdict(float)
    for n in coupling:
        combined[n.doc_id] += coupling_weight * n.score
    for n in cocitation:
        combined[n.doc_id] += cocitation_weight * n.score

    ranked = sorted(
        ((doc_id, score) for doc_id, score in combined.items() if score > threshold),
        key=lambda t: t[1],
        reverse=True,
    )[:top_n]

    nodes = [seed_doc_id] + [doc_id for doc_id, _ in ranked]
    edges = [
        {"src": seed_doc_id, "dst": doc_id, "weight": round(score, 6)} for doc_id, score in ranked
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": seed_doc_id,
        "nodes": nodes,
        "edges": edges,
    }


def to_dot(graph: dict) -> str:
    """Render a `neighborhood_graph` (or any `{"nodes": [...], "edges": [...]}` payload) as
    a Graphviz DOT string."""

    def _q(s: str) -> str:
        return '"' + s.replace('"', '\\"') + '"'

    lines = ["graph G {"]
    for node in graph.get("nodes", []):
        lines.append(f"  {_q(node)};")
    for edge in graph.get("edges", []):
        weight = edge.get("weight", 1.0)
        lines.append(f"  {_q(edge['src'])} -- {_q(edge['dst'])} [weight={weight}];")
    lines.append("}")
    return "\n".join(lines)
