"""Offline unit tests for the HippoRAG-2-style graph retrieval arm.

No GPU / no model loads / no index: retrieval seeds are mocked and the graph is a tiny
synthetic adjacency. Covers the PPR/blend math, graph construction, kNN cache versioning,
and result-row-shape compatibility with ``query.search.format_results``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from graph_layout_rag.query import graph_arm as ga


# --------------------------------------------------------------------------- #
# Seed scoring
# --------------------------------------------------------------------------- #
def _canon(row):
    return row["doc_id"]


def test_seed_doc_scores_takes_max_per_doc() -> None:
    rows = [
        {"doc_id": "a", "fusion_score": 0.1},
        {"doc_id": "a", "fusion_score": 0.3},
        {"doc_id": "b", "score": 0.2},
    ]
    assert ga.seed_doc_scores(rows, _canon) == {"a": 0.3, "b": 0.2}


def test_normalize_restart_is_a_distribution() -> None:
    dist = ga.normalize_restart({"a": 3.0, "b": 1.0})
    assert dist["a"] == pytest.approx(0.75)
    assert dist["b"] == pytest.approx(0.25)
    assert sum(dist.values()) == pytest.approx(1.0)


def test_normalize_restart_uniform_when_nonpositive() -> None:
    dist = ga.normalize_restart({"a": 0.0, "b": 0.0})
    assert dist == {"a": 0.5, "b": 0.5}


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
def test_knn_edges_symmetric_and_clamped() -> None:
    # 3 unit vectors: a,b close; c orthogonal.
    vectors = {
        "a": [1.0, 0.0],
        "b": [0.99, 0.14],
        "c": [0.0, 1.0],
    }
    edges = ga.knn_edges(vectors, k=2)
    # a-b edge exists both directions with equal (max) weight, clamped >= 0.
    assert edges["a"]["b"] == edges["b"]["a"]
    assert edges["a"]["b"] > 0.5
    assert all(w >= 0.0 for nbrs in edges.values() for w in nbrs.values())


def test_citation_doc_edges_projects_in_corpus_only() -> None:
    @dataclass
    class FakeGraph:
        oa_to_doc: dict = field(default_factory=dict)
        undirected: dict = field(default_factory=dict)

    g = FakeGraph(
        oa_to_doc={"W1": "a", "W2": "b"},  # W3 is a non-corpus reference
        undirected={"W1": {"W2", "W3"}, "W2": {"W1"}, "W3": {"W1"}},
    )
    edges = ga.citation_doc_edges(g)
    assert edges == {"a": {"b"}, "b": {"a"}}  # W3 dropped (no corpus doc)


def test_citation_doc_edges_none_graph() -> None:
    assert ga.citation_doc_edges(None) == {}


def test_union_graph_sums_overlapping_signals() -> None:
    knn = {"a": {"b": 0.5}, "b": {"a": 0.5}}
    cite = {"a": {"b"}, "b": {"a"}}
    adj = ga.union_graph(knn, cite, knn_weight=1.0, cite_weight=1.0)
    assert adj["a"]["b"] == pytest.approx(1.5)  # 0.5 kNN + 1.0 citation
    assert adj["a"]["b"] == adj["b"]["a"]


# --------------------------------------------------------------------------- #
# Weighted personalized PageRank
# --------------------------------------------------------------------------- #
def test_seeded_pagerank_conserves_mass_and_favors_seed() -> None:
    # path graph a-b-c, all restart mass on a.
    adj = {
        "a": {"b": 1.0},
        "b": {"a": 1.0, "c": 1.0},
        "c": {"b": 1.0},
    }
    rank = ga.seeded_pagerank(adj, {"a": 1.0}, restart=0.5)
    assert sum(rank.values()) == pytest.approx(1.0, abs=1e-6)
    assert rank["a"] > rank["b"] > rank["c"] > 0.0


def test_seeded_pagerank_weighted_teleport_biases_toward_heavier_seed() -> None:
    adj = {"a": {"x": 1.0}, "b": {"y": 1.0}, "x": {"a": 1.0}, "y": {"b": 1.0}}
    rank = ga.seeded_pagerank(adj, {"a": 3.0, "b": 1.0}, restart=0.5)
    assert rank["a"] > rank["b"]
    assert rank["x"] > rank["y"]  # a's neighbor inherits more mass than b's


def test_seeded_pagerank_empty() -> None:
    assert ga.seeded_pagerank({}, {}) == {}


# --------------------------------------------------------------------------- #
# Blend math (PPR + hybrid)
# --------------------------------------------------------------------------- #
def test_blend_alpha_zero_is_pure_hybrid() -> None:
    ppr = {"a": 1.0, "b": 0.1}
    hybrid = {"a": 0.2, "b": 0.8}
    blended = ga.blend_scores(ppr, hybrid, alpha=0.0)
    # alpha=0 -> normalized hybrid ranking exactly (b > a).
    assert blended["b"] > blended["a"]
    assert blended["b"] == pytest.approx(1.0)


def test_blend_alpha_one_is_pure_ppr() -> None:
    ppr = {"a": 1.0, "b": 0.1}
    hybrid = {"a": 0.2, "b": 0.8}
    blended = ga.blend_scores(ppr, hybrid, alpha=1.0)
    assert blended["a"] > blended["b"]
    assert blended["a"] == pytest.approx(1.0)


def test_blend_midpoint_is_average_of_normalized() -> None:
    ppr = {"a": 2.0}  # normalizes to 1.0
    hybrid = {"a": 4.0}  # normalizes to 1.0
    assert ga.blend_scores(ppr, hybrid, alpha=0.5)["a"] == pytest.approx(1.0)


def test_blend_missing_signal_contributes_zero() -> None:
    blended = ga.blend_scores({"a": 1.0}, {"b": 1.0}, alpha=0.5)
    assert blended["a"] == pytest.approx(0.5)  # only ppr
    assert blended["b"] == pytest.approx(0.5)  # only hybrid


def test_blend_rejects_bad_alpha() -> None:
    with pytest.raises(ValueError):
        ga.blend_scores({}, {}, alpha=1.5)


# --------------------------------------------------------------------------- #
# Result-row assembly (shape compatible with format_results)
# --------------------------------------------------------------------------- #
def test_assemble_rows_orders_by_blend_and_sets_fusion_score() -> None:
    pool = {
        "a": [{"doc_id": "a", "id": "a:0", "rerank_score": 9.9, "text": "x"}],
        "b": [{"doc_id": "b", "id": "b:0", "text": "y"}],
    }
    blended = {"a": 0.9, "b": 0.4}
    rows = ga.assemble_rows(["a", "b"], blended, pool, fetch_fn=lambda d: [], top=8, max_per_doc=2)
    assert [r["doc_id"] for r in rows] == ["a", "b"]
    # rerank_score dropped so downstream _score() reports the blended fusion_score.
    assert "rerank_score" not in rows[0]
    assert rows[0]["fusion_score"] == pytest.approx(0.9)
    assert rows[1]["fusion_score"] == pytest.approx(0.4)


def test_assemble_rows_fetches_out_of_pool_docs() -> None:
    pool = {"a": [{"doc_id": "a", "id": "a:0", "text": "x"}]}
    blended = {"a": 0.9, "z": 0.8}  # z surfaced only by the graph walk

    def fetch(doc):
        return [{"doc_id": doc, "id": f"{doc}:0", "text": "fetched"}] if doc == "z" else []

    rows = ga.assemble_rows(["a", "z"], blended, pool, fetch_fn=fetch, top=8, max_per_doc=2)
    assert [r["doc_id"] for r in rows] == ["a", "z"]
    assert rows[1]["fusion_score"] == pytest.approx(0.8)


def test_assemble_rows_respects_top_and_max_per_doc() -> None:
    pool = {
        "a": [
            {"doc_id": "a", "id": "a:0", "text": "1"},
            {"doc_id": "a", "id": "a:1", "text": "2"},
            {"doc_id": "a", "id": "a:2", "text": "3"},
        ],
        "b": [{"doc_id": "b", "id": "b:0", "text": "y"}],
    }
    rows = ga.assemble_rows(
        ["a", "b"], {"a": 0.9, "b": 0.4}, pool, fetch_fn=lambda d: [], top=1, max_per_doc=2
    )
    # top=1 -> only doc a; max_per_doc=2 -> only 2 of its 3 chunks.
    assert {r["doc_id"] for r in rows} == {"a"}
    assert len(rows) == 2


# --------------------------------------------------------------------------- #
# kNN adjacency cache versioning
# --------------------------------------------------------------------------- #
def test_knn_cache_roundtrip_and_version_key(tmp_path) -> None:
    path = tmp_path / "graph_arm_knn.json"
    edges = {"a": {"b": 0.5}, "b": {"a": 0.5}}
    ga._write_knn_cache(path, edges, profile="p1", fingerprint="fp1", k=15)

    # exact version key -> hit
    hit = ga._read_knn_cache(path, profile="p1", fingerprint="fp1", k=15)
    assert hit == edges

    # any component of the version key mismatches -> miss (None)
    assert ga._read_knn_cache(path, profile="p2", fingerprint="fp1", k=15) is None
    assert ga._read_knn_cache(path, profile="p1", fingerprint="fp2", k=15) is None
    assert ga._read_knn_cache(path, profile="p1", fingerprint="fp1", k=10) is None


def test_knn_cache_missing_file_is_miss(tmp_path) -> None:
    path = tmp_path / "absent.json"
    assert ga._read_knn_cache(path, profile="p", fingerprint="f", k=15) is None
