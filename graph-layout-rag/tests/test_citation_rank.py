"""Toy-graph tests for the relatedness math (no network, no DB)."""

from collections import defaultdict

from graph_layout_rag.query.citation_rank import (
    CitationGraph,
    bibliographic_coupling,
    co_citation,
    in_corpus_citation_stats,
    personalized_pagerank,
    rank_related,
    related_to_docs,
)
from graph_layout_rag.query.identity import CanonicalIdentityMap


def _toy() -> CitationGraph:
    # Corpus A,B,C cite external refs R1,R2; external X cites A and B.
    #   A -> R1, R2 ; B -> R1 ; C -> R2 ; X -> A, B
    g = CitationGraph()
    edges = [("Wa", "R1"), ("Wa", "R2"), ("Wb", "R1"), ("Wc", "R2"), ("Wx", "Wa"), ("Wx", "Wb")]
    for s, d in edges:
        g.out_adj[s].add(d)
        g.in_adj[d].add(s)
        g.undirected[s].add(d)
        g.undirected[d].add(s)
    g.cbc = {"Wa": 100, "Wb": 10, "Wc": 5, "Wx": 1, "R1": 50, "R2": 40}
    g.oa_to_doc = {"Wa": "doc-a", "Wb": "doc-b", "Wc": "doc-c"}
    g.doc_to_oa = {v: k for k, v in g.oa_to_doc.items()}
    return g


def test_bibliographic_coupling():
    g = _toy()
    seed_refs = g.out_adj["Wa"]  # {R1, R2}
    assert bibliographic_coupling(g, seed_refs, "Wb") > 0  # shares R1
    assert bibliographic_coupling(g, seed_refs, "Wc") > 0  # shares R2
    assert bibliographic_coupling(g, seed_refs, "Wx") == 0  # shares none


def test_co_citation():
    g = _toy()
    seed_citers = g.in_adj["Wa"]  # {Wx}
    assert co_citation(g, seed_citers, "Wb") > 0  # both cited by Wx
    assert co_citation(g, seed_citers, "Wc") == 0


def test_ppr_seed_is_top_and_spreads():
    g = _toy()
    ppr = personalized_pagerank(g, {"Wa"}, iters=100)
    assert ppr  # non-empty
    assert max(ppr, key=ppr.get) == "Wa"  # restart node dominates
    assert ppr.get("Wb", 0) > 0  # reachable via R1 and via Wx
    assert ppr.get("Wc", 0) > 0  # reachable via R2
    # mass is a proper distribution-ish (bounded, positive)
    assert all(v >= 0 for v in ppr.values())


def test_ppr_empty_for_unknown_seed():
    g = _toy()
    assert personalized_pagerank(g, {"W-missing"}) == {}


def test_rank_related_returns_corpus_neighbors():
    g = _toy()
    ranked = rank_related(g, {"Wa"}, {"Wb", "Wc", "Wx"})
    docs = [r.doc_id for r in ranked]
    assert "doc-b" in docs and "doc-c" in docs  # both related to A
    # Wb shares a reference AND a co-citation with Wa; Wc only a reference -> Wb ranks first.
    assert docs[0] == "doc-b"
    top = ranked[0]
    assert top.shared_refs >= 1 and top.shared_citations >= 1


def test_related_to_docs_resolves_seed_alias_and_deduplicates_results(monkeypatch):
    g = _toy()
    g.oa_to_doc["Wb2"] = "doc-b-alias"
    g.doc_to_oa["doc-b-alias"] = "Wb2"
    g.add_edge("Wb2", "R1")
    identities = CanonicalIdentityMap(
        canonical_by_doc={
            "doc-a": "doc-a",
            "doc-a-alias": "doc-a",
            "doc-b": "doc-b",
            "doc-b-alias": "doc-b",
            "doc-c": "doc-c",
        },
        aliases_by_canonical={
            "doc-a": ("doc-a-alias",),
            "doc-b": ("doc-b-alias",),
            "doc-c": (),
        },
    )
    monkeypatch.setattr(
        "graph_layout_rag.query.identity.canonical_identity_map",
        lambda: identities,
    )

    ranked = related_to_docs(None, ["doc-a-alias"], graph=g, top=10)
    assert ranked
    assert [result.doc_id for result in ranked].count("doc-b") == 1


def _stats_graph() -> CitationGraph:
    # Corpus Wa,Wb,Wc; external Wx,Wy. Edges:
    #   Wa -> Wb (corpus ref), Wa -> R1 (non-corpus ref)
    #   Wb -> Wa (corpus citer of Wa), Wx -> Wa (non-corpus citer), Wa -> Wa (self, ignored)
    g = CitationGraph()
    g.oa_to_doc = {"Wa": "doc-a", "Wb": "doc-b", "Wc": "doc-c"}
    g.doc_to_oa = {v: k for k, v in g.oa_to_doc.items()}
    g.cbc = {"Wa": 42, "Wb": 3, "Wc": 0}
    for s, d in [("Wa", "Wb"), ("Wa", "R1"), ("Wb", "Wa"), ("Wx", "Wa"), ("Wa", "Wa")]:
        g.out_adj[s].add(d)
        g.in_adj[d].add(s)
        g.undirected[s].add(d)
        g.undirected[d].add(s)
    return g


def test_in_corpus_stats_no_graph_returns_empty(monkeypatch):
    # graph=None with no store -> load_graph_cached returns None -> {}
    monkeypatch.setattr(
        "graph_layout_rag.query.citation_rank.load_graph_cached", lambda: None
    )
    assert in_corpus_citation_stats("doc-a", None) == {}


def test_in_corpus_stats_doc_without_oa_node():
    g = _stats_graph()
    stats = in_corpus_citation_stats("doc-missing", g)
    assert stats == {
        "cited_by_count": None,
        "in_corpus_cited_by_count": 0,
        "in_corpus_references_count": 0,
    }


def test_in_corpus_stats_counts_only_corpus_endpoints_and_excludes_self():
    g = _stats_graph()
    stats = in_corpus_citation_stats("doc-a", g)
    assert stats["cited_by_count"] == 42  # global
    # citers of Wa: Wb (corpus), Wx (non-corpus), Wa (self) -> only Wb counts
    assert stats["in_corpus_cited_by_count"] == 1
    # refs of Wa: Wb (corpus), R1 (non-corpus), Wa (self) -> only Wb counts
    assert stats["in_corpus_references_count"] == 1


def test_in_corpus_stats_zero_in_corpus_citers():
    g = _stats_graph()
    stats = in_corpus_citation_stats("doc-c", g)
    assert stats["cited_by_count"] == 0
    assert stats["in_corpus_cited_by_count"] == 0
    assert stats["in_corpus_references_count"] == 0


def test_load_graph_bridges_doc_id_by_doi(tmp_path, monkeypatch):
    """DOI bridge maps a DOI-only manifest id (no papers.doc_id) to its oa."""
    import graph_layout_rag.citation_store as cs
    from graph_layout_rag.manifest import Manifest, ManifestItem
    from graph_layout_rag.query.citation_rank import load_graph

    path = tmp_path / "citations.sqlite"
    monkeypatch.setattr(cs, "CITATIONS_DB_PATH", path)
    db = cs.connect(path)
    # Wa is seeded with a doc_id (direct); Wb has only a DOI (cite_enrich never
    # wrote its doc_id), so it must be recovered via the DOI bridge.
    db.execute(
        "INSERT INTO papers(oa_id, doc_id, doi, cited_by_count, in_corpus) VALUES (?,?,?,?,?)",
        ("Wa", "direct-1", "10.1/a", 100, 1),
    )
    db.execute(
        "INSERT INTO papers(oa_id, doc_id, doi, cited_by_count, in_corpus) VALUES (?,?,?,?,?)",
        ("Wb", None, "10.1/b", 7, 0),
    )
    db.commit()

    items = [
        ManifestItem(id="direct-1", title="A", source="s", url="u", status="ok", doi="10.1/a"),
        ManifestItem(id="bridged-2", title="B", source="s", url="u", status="ok", doi="10.1/B"),
        ManifestItem(id="nodoi-3", title="C", source="s", url="u", status="ok"),
    ]
    monkeypatch.setattr(
        "graph_layout_rag.manifest.load_manifest",
        lambda: Manifest(items=items),
    )

    g = load_graph(db)
    db.close()
    assert g.doc_to_oa.get("direct-1") == "Wa"          # direct mapping preserved
    assert g.doc_to_oa.get("bridged-2") == "Wb"         # recovered via DOI (case-normalized)
    assert "nodoi-3" not in g.doc_to_oa                  # no DOI -> not bridged
    assert g.oa_to_doc.get("Wb") == "bridged-2"
