"""Shared E1 map-quality eval library: reference-concentration metric, hybrid-vs-single-
signal ablation gate, survey gold-set builder, and judge calibration harness.

Tool-agnostic (no `graph_layout_rag`/`rag_literature_rag` import) — each tool's `cli.py`
passes in its own citation DB connection / manifest / doc vectors, mirroring the seam
already used by `rag_common.litmap` (`CitationAdjacency.from_graph`) and
`rag_common.citation_coverage`.
"""

from __future__ import annotations
