"""Shared plain-data types for the litmap layer.

`CitationAdjacency` is the dependency-injection seam: each tool's `query/citation_rank.py`
owns a `CitationGraph` dataclass (out_adj/in_adj/undirected/doc_to_oa/oa_to_doc) loaded from
its own `citations.sqlite`. litmap never imports a tool package, so callers convert their
loaded graph into this duck-typed shape once per CLI invocation via `from_graph`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Neighbor:
    """One ranked neighbor from a single similarity signal or a fused result."""

    doc_id: str
    rank: int
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, "rank": self.rank, "score": round(self.score, 6)}


@dataclass(frozen=True)
class CitationAdjacency:
    """Duck-typed view of a tool's citation graph, keyed by OpenAlex-id-space adjacency
    plus the doc_id<->oa_id maps needed to translate to/from corpus doc ids."""

    out_adj: Mapping[str, set[str]] = field(default_factory=dict)  # references (oa -> oa)
    in_adj: Mapping[str, set[str]] = field(default_factory=dict)  # cited-by (oa -> oa)
    undirected: Mapping[str, set[str]] = field(default_factory=dict)
    doc_to_oa: Mapping[str, str] = field(default_factory=dict)
    oa_to_doc: Mapping[str, str] = field(default_factory=dict)
    years: Mapping[str, int] = field(default_factory=dict)  # oa_id -> year, optional

    @classmethod
    def from_graph(cls, graph: Any) -> CitationAdjacency:
        """Build from any object exposing the `CitationGraph` attributes (duck typing —
        avoids importing either tool's package from this shared library)."""
        return cls(
            out_adj=getattr(graph, "out_adj", {}),
            in_adj=getattr(graph, "in_adj", {}),
            undirected=getattr(graph, "undirected", {}),
            doc_to_oa=getattr(graph, "doc_to_oa", {}),
            oa_to_doc=getattr(graph, "oa_to_doc", {}),
            years=getattr(graph, "years", {}),
        )

    def idf(self, oa_id: str) -> float:
        """Rare shared neighbors are more informative (same formula as
        `query/citation_rank._idf`, duplicated here since it's a 2-line closed formula
        over injected adjacency, not worth an import)."""
        import math

        deg = len(self.undirected.get(oa_id, ()))
        return 1.0 / math.log2(2.0 + deg)
