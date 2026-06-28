"""format_results metadata join: positive surfacing + graceful-degrade.

These never hit the index/DB — they drive ``format_results`` directly with a
synthetic reranked row and monkeypatch the three metadata sources.
"""

from __future__ import annotations

import pytest

from graph_layout_rag.citation_store import PaperMeta
from graph_layout_rag.manifest import ManifestItem
from graph_layout_rag.query import search as search_mod
from graph_layout_rag.query.citation_rank import CitationGraph
from graph_layout_rag.query.search import format_results


def _row(doc_id: str = "doc-a") -> dict:
    return {
        "doc_id": doc_id,
        "title": "Layered Graph Drawing",
        "text": "Sugiyama-style layered layout assigns nodes to discrete ranks.",
        "source_url": "https://example.org/a",
        "id": "chunk-1",
        "score": 1.0,
    }


@pytest.fixture(autouse=True)
def _no_catalog_or_identity(monkeypatch):
    # Keep format_results' pre-existing dependencies inert so the test is hermetic.
    monkeypatch.setattr(search_mod, "catalog_maps", lambda: ({}, {}))

    class _Ident:
        def canonical_doc_id(self, row):
            return row.get("doc_id") or ""

        def aliases(self, canonical, row):
            return []

    monkeypatch.setattr(search_mod, "canonical_identity_map", lambda: _Ident())
    monkeypatch.setattr(search_mod, "split_values", lambda v: [])


def _graph() -> CitationGraph:
    g = CitationGraph()
    g.oa_to_doc = {"Wa": "doc-a", "Wb": "doc-b"}
    g.doc_to_oa = {"doc-a": "Wa", "doc-b": "Wb"}
    g.cbc = {"Wa": 99, "Wb": 1}
    for s, d in [("Wb", "Wa")]:  # doc-b cites doc-a (in-corpus citer)
        g.out_adj[s].add(d)
        g.in_adj[d].add(s)
        g.undirected[s].add(d)
        g.undirected[d].add(s)
    return g


def test_join_surfaces_meta_citation_and_bibtex(monkeypatch):
    meta = PaperMeta(
        doc_id="doc-a",
        tldr="Layered layout in linear time.",
        fwci=2.5,
        oa_pdf_url="https://example.org/a.pdf",
    )
    monkeypatch.setattr(
        "graph_layout_rag.citation_store.load_paper_meta_cached",
        lambda: {"doc-a": meta},
    )
    monkeypatch.setattr(
        "graph_layout_rag.query.citation_rank.load_graph_cached",
        lambda: _graph(),
    )
    item = ManifestItem(
        id="doc-a",
        title="Layered Graph Drawing",
        authors=["Emden Gansner"],
        year=1993,
        source="arxiv",
        url="https://example.org/a",
        status="ok",
        venue="Graph Drawing Symposium",
        genre="article",
    )
    monkeypatch.setattr(search_mod, "_manifest_by_doc", lambda: {"doc-a": item})

    out = format_results([_row("doc-a")], top=5, max_per_doc=2)
    assert len(out) == 1
    entry = out[0]
    assert entry["tldr"] == "Layered layout in linear time."
    assert entry["fwci"] == 2.5
    assert entry["oa_pdf_url"] == "https://example.org/a.pdf"
    assert entry["cited_by_count"] == 99
    assert entry["in_corpus_cited_by_count"] == 1
    assert entry["venue"] == "Graph Drawing Symposium"
    assert entry["genre"] == "article"
    assert entry["is_retracted"] is False
    assert entry["bibtex"].startswith("@inproceedings{gansner1993layered,")


def test_graceful_degrade_when_all_sources_empty(monkeypatch):
    monkeypatch.setattr(
        "graph_layout_rag.citation_store.load_paper_meta_cached", lambda: {}
    )
    monkeypatch.setattr(
        "graph_layout_rag.query.citation_rank.load_graph_cached", lambda: None
    )
    monkeypatch.setattr(search_mod, "_manifest_by_doc", lambda: {})

    out = format_results([_row("doc-absent")], top=5, max_per_doc=2)
    assert len(out) == 1
    entry = out[0]
    # No metadata keys added, no exception, existing fields intact.
    for key in (
        "tldr",
        "fwci",
        "oa_pdf_url",
        "cited_by_count",
        "in_corpus_cited_by_count",
        "venue",
        "genre",
        "is_retracted",
        "bibtex",
    ):
        assert key not in entry
    assert entry["title"] == "Layered Graph Drawing"
    assert entry["doc_id"] == "doc-absent"
