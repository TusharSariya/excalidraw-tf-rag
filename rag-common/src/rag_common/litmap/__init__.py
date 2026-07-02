"""Knowledge-map layer (E1): landscape mapping over a corpus's citation graph + doc vectors.

Tool-agnostic by construction — every function here takes plain data (adjacency dicts,
vector maps, edge lists) injected by the caller, never a tool-specific import. Each tool
(`graph-layout-rag`, `rag-literature-rag`) owns the citation-graph loading
(`query/citation_rank.py`) and the champion LanceDB index; its `cli.py` wires those into
these functions and renders the results.

Modules:
  * `doc_vectors` — chunk-count-weighted mean-pooled, L2-normalized doc vectors from a
    LanceDB chunk index, persisted as npz + json meta.
  * `similarity` — per-signal top-k neighbor lists (bibliographic coupling, cosine kNN)
    fused by rank (RRF), not raw score.
  * `communities` — Leiden partition of the fused kNN graph, with extractive (default) or
    LLM-assisted cluster labels.
  * `mainpath` — main-path (SPC) analysis over a citation DAG, with temporal validation
    and cycle-breaking first.
  * `neighborhood` — Connected-Papers-style local similarity neighborhood around a seed doc.

All public functions return JSON-serializable dicts with a `"schema_version": 1` key.
"""

from __future__ import annotations

SCHEMA_VERSION = 1
