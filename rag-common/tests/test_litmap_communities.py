from __future__ import annotations

import pytest
from rag_common.litmap import communities as com

igraph = pytest.importorskip("igraph")
pytest.importorskip("leidenalg")


def test_leiden_two_cliques_two_clusters():
    # two disjoint 4-cliques, densely connected within, no cross edges -> Leiden should
    # find (at least) two separate clusters and never merge them.
    edges = []
    clique_a = ["a1", "a2", "a3", "a4"]
    clique_b = ["b1", "b2", "b3", "b4"]
    for i in range(len(clique_a)):
        for j in range(i + 1, len(clique_a)):
            edges.append((clique_a[i], clique_a[j], 1.0))
    for i in range(len(clique_b)):
        for j in range(i + 1, len(clique_b)):
            edges.append((clique_b[i], clique_b[j], 1.0))

    membership = com.leiden_communities(edges, seed=42)
    assert membership[clique_a[0]] != membership[clique_b[0]]
    assert len({membership[n] for n in clique_a}) == 1
    assert len({membership[n] for n in clique_b}) == 1


def test_leiden_seed_determinism():
    edges = [("a", "b", 1.0), ("b", "c", 1.0), ("c", "a", 1.0), ("d", "e", 1.0), ("e", "f", 1.0)]
    m1 = com.leiden_communities(edges, seed=7)
    m2 = com.leiden_communities(edges, seed=7)
    assert m1 == m2


def test_leiden_disconnected_graph_gets_per_component_partition():
    # a single edge component plus two fully isolated docs (no edges at all, supplied via
    # all_doc_ids) -> every doc still gets an id, isolated docs are singleton clusters.
    edges = [("a", "b", 1.0)]
    membership = com.leiden_communities(edges, all_doc_ids=["a", "b", "isolated1", "isolated2"])
    assert set(membership) == {"a", "b", "isolated1", "isolated2"}
    assert membership["isolated1"] != membership["isolated2"]
    assert membership["a"] == membership["b"]


def test_leiden_empty_input():
    assert com.leiden_communities([]) == {}


def test_extractive_label_favors_distinctive_terms():
    # cluster is all about "graphviz layout"; corpus at large is mostly generic RAG text,
    # so "graphviz"/"layout" should out-rank generic corpus-wide terms in the label.
    corpus_texts = (
        ["graphviz layout algorithm"] * 3
        + ["dense retrieval embedding model"] * 20
        + ["retrieval augmented generation survey"] * 20
    )
    corpus_df, corpus_size = com.build_corpus_df(corpus_texts)
    label = com.extractive_label(
        ["graphviz layout algorithm", "graphviz sugiyama layout"],
        corpus_df,
        corpus_size,
        top_terms=2,
    )
    assert "graphviz" in label
    assert "layout" in label


def test_extractive_label_empty_cluster_is_unlabeled():
    assert com.extractive_label([], {}, 0) == "unlabeled cluster"


def test_llm_label_failure_falls_back_to_extractive(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("backend unreachable")

    monkeypatch.setattr("rag_common.local_llm.generate_text", _boom)

    corpus_df, corpus_size = com.build_corpus_df(["graphviz layout paper"])
    label, method = com.llm_label(
        ["Graphviz Layout Paper"],
        fallback_texts=["graphviz layout paper"],
        corpus_df=corpus_df,
        corpus_size=corpus_size,
    )
    assert method == "extractive"
    assert label != ""


def test_llm_label_success_path(monkeypatch):
    monkeypatch.setattr("rag_common.local_llm.generate_text", lambda *a, **k: "Graph Layout Theory")
    label, method = com.llm_label(
        ["Some Title"], fallback_texts=["some title"], corpus_df={}, corpus_size=1
    )
    assert method == "llm"
    assert label == "Graph Layout Theory"


def test_llm_label_empty_response_falls_back(monkeypatch):
    monkeypatch.setattr("rag_common.local_llm.generate_text", lambda *a, **k: "   ")
    corpus_df, corpus_size = com.build_corpus_df(["graphviz layout paper"])
    label, method = com.llm_label(
        ["Graphviz Layout Paper"],
        fallback_texts=["graphviz layout paper"],
        corpus_df=corpus_df,
        corpus_size=corpus_size,
    )
    assert method == "extractive"


def test_build_clusters_schema():
    edges = [("a", "b", 1.0)]
    doc_meta = {
        "a": {"title": "Graphviz Layout", "tags": ["layout"]},
        "b": {"title": "Sugiyama Method", "tags": ["layout"]},
    }
    result = com.build_clusters(edges, doc_meta)
    assert result["schema_version"] == 1
    assert "clusters" in result
    assert result["doc_count"] == 2
    for cluster in result["clusters"]:
        for key in ("id", "size", "label", "label_method", "doc_ids"):
            assert key in cluster
