"""Research-tool navigation features (citation-data value beyond the NULL ranking
prior): explicit citation sort, the 'seminal' entry-point flag, and in-corpus
citation-neighborhood traversal. All are user-driven surfacing — none change the
relevance score."""
from __future__ import annotations

from graph_layout_rag.query.citation_rank import CitationGraph, in_corpus_neighbors
from graph_layout_rag.query.search import _flag_seminal, sort_results


def _entries():
    return [
        {"doc_id": "a", "cited_by_count": 5, "in_corpus_cited_by_count": 1},
        {"doc_id": "b", "cited_by_count": 50, "in_corpus_cited_by_count": 0},
        {"doc_id": "c", "in_corpus_cited_by_count": 9},  # no global count
    ]


# --------------------------------------------------------------------------- sort
def test_sort_relevance_is_passthrough():
    e = _entries()
    assert sort_results(e, "relevance") is e
    assert sort_results(e, None) is e


def test_sort_by_global_cited_by():
    order = [x["doc_id"] for x in sort_results(_entries(), "cited-by")]
    assert order == ["b", "a", "c"]  # 50, 5, 0(missing)


def test_sort_by_in_corpus_cited_by():
    order = [x["doc_id"] for x in sort_results(_entries(), "in-corpus-cited-by")]
    assert order == ["c", "a", "b"]  # 9, 1, 0


def test_sort_is_stable_on_ties():
    e = [
        {"doc_id": "x", "cited_by_count": 7},
        {"doc_id": "y", "cited_by_count": 7},
        {"doc_id": "z", "cited_by_count": 7},
    ]
    assert [r["doc_id"] for r in sort_results(e, "cited-by")] == ["x", "y", "z"]


# --------------------------------------------------------------------------- seminal
def test_flag_seminal_marks_max_in_corpus():
    e = _entries()
    _flag_seminal(e)
    flagged = [x["doc_id"] for x in e if x.get("seminal")]
    assert flagged == ["c"]  # highest in_corpus_cited_by_count (9)


def test_flag_seminal_noop_when_all_zero():
    e = [{"doc_id": "a"}, {"doc_id": "b", "in_corpus_cited_by_count": 0}]
    _flag_seminal(e)
    assert not any(x.get("seminal") for x in e)


# --------------------------------------------------------------------------- neighbors
def _graph():
    g = CitationGraph()
    # Wa references R1,R2 (builds-on) and is cited by Wx and a non-corpus node N
    g.out_adj["Wa"] = {"R1", "R2", "Wa"}  # includes a self-loop to be ignored
    g.in_adj["Wa"] = {"Wx", "N", "Wa"}
    g.cbc = {"R1": 50, "R2": 40, "Wx": 5}
    g.oa_to_doc = {"Wa": "doc-a", "R1": "doc-r1", "R2": "doc-r2", "Wx": "doc-x"}
    g.doc_to_oa = {v: k for k, v in g.oa_to_doc.items()}
    return g


def test_neighbors_builds_on_sorted_and_corpus_only():
    nb = in_corpus_neighbors("doc-a", _graph())
    builds = [(n["doc_id"], n["cited_by_count"]) for n in nb["builds_on"]]
    assert builds == [("doc-r1", 50), ("doc-r2", 40)]  # sorted desc; self-loop dropped


def test_neighbors_cited_by_excludes_non_corpus_and_self():
    nb = in_corpus_neighbors("doc-a", _graph())
    assert [n["doc_id"] for n in nb["cited_by"]] == ["doc-x"]  # 'N' non-corpus, 'Wa' self dropped


def test_neighbors_limit():
    nb = in_corpus_neighbors("doc-a", _graph(), limit=1)
    assert len(nb["builds_on"]) == 1 and nb["builds_on"][0]["doc_id"] == "doc-r1"


def test_neighbors_unknown_doc_is_zeroed():
    assert in_corpus_neighbors("nope", _graph()) == {"builds_on": [], "cited_by": []}


def test_neighbors_no_graph_returns_empty(monkeypatch):
    import graph_layout_rag.query.citation_rank as cr

    monkeypatch.setattr(cr, "load_graph_cached", lambda: None)
    assert cr.in_corpus_neighbors("doc-a", None) == {}
