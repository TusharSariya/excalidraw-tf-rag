"""T7: optional citation/recency ranking prior blended into RRF fusion.

The headline safety contract is the *default-OFF byte-identical* guard: at
``citation_prior_weight=0.0`` (the default) the fused ranking — order AND every
``fusion_score`` value — must match the pre-T7 path exactly. These tests pin
that guard, plus the bounded behaviour and graceful neutrality when the weight
is on but signals (graph node / year) are missing.
"""

from __future__ import annotations

import copy

from graph_layout_rag.query.citation_rank import CitationGraph
from graph_layout_rag.query.hybrid import reciprocal_rank_fusion


def _dense() -> list[dict]:
    return [
        {"id": "docA:0", "doc_id": "docA", "score": 0.9, "year": 2021},
        {"id": "docB:0", "doc_id": "docB", "score": 0.8, "year": 2010},
        {"id": "docC:0", "doc_id": "docC", "score": 0.7, "year": 1998},
    ]


def _sparse() -> list[dict]:
    return [
        {"id": "docB:0", "doc_id": "docB", "score": 12.0, "year": 2010},
        {"id": "docC:0", "doc_id": "docC", "score": 9.0, "year": 1998},
        {"id": "docD:0", "doc_id": "docD", "score": 4.0, "year": 2024},
    ]


def _graph() -> CitationGraph:
    g = CitationGraph()
    g.doc_to_oa = {"docA": "oaA", "docB": "oaB", "docC": "oaC", "docD": "oaD"}
    g.oa_to_doc = {v: k for k, v in g.doc_to_oa.items()}
    # docA is the highest-cited paper of all, but it sits at the BOTTOM of the
    # fusion ranking. docC is a fusion-adjacent (#2) high-cited paper.
    g.cbc = {"oaA": 100000, "oaB": 20, "oaC": 80000, "oaD": 5}
    return g


def test_weight_zero_is_byte_identical_to_no_prior_params() -> None:
    """CRITICAL regression guard: weight 0.0 == calling without the new params."""
    baseline = reciprocal_rank_fusion(_dense(), _sparse())
    with_param = reciprocal_rank_fusion(
        _dense(), _sparse(), citation_prior_weight=0.0, citation_graph=_graph()
    )
    assert baseline == with_param
    # order + exact fusion_score values
    assert [r["id"] for r in baseline] == [r["id"] for r in with_param]
    assert [r["fusion_score"] for r in baseline] == [r["fusion_score"] for r in with_param]


def test_graph_none_with_positive_weight_is_identical() -> None:
    baseline = reciprocal_rank_fusion(_dense(), _sparse())
    out = reciprocal_rank_fusion(
        _dense(), _sparse(), citation_prior_weight=0.05, citation_graph=None
    )
    assert baseline == out


def test_positive_weight_improves_high_citation_rank_but_stays_bounded() -> None:
    weight = 0.02
    baseline = reciprocal_rank_fusion(_dense(), _sparse())
    base_rank = {r["id"]: i for i, r in enumerate(baseline)}
    base_scores = {r["id"]: r["fusion_score"] for r in baseline}
    # Sanity: docA is genuinely the bottom of the fusion ranking.
    assert baseline[-1]["id"] == "docA:0"

    boosted = reciprocal_rank_fusion(
        _dense(), _sparse(), citation_prior_weight=weight, citation_graph=_graph()
    )
    boost_rank = {r["id"]: i for i, r in enumerate(boosted)}

    # A high-citation, fusion-adjacent doc (docC, #2) climbs in rank.
    assert boost_rank["docC:0"] < base_rank["docC:0"]

    # BOUNDED: docA has the single highest cited-by count of all, but because it
    # sits at the bottom of the fusion ranking, a small weight must NOT let the
    # normalized prior catapult it to #1.
    assert boost_rank["docA:0"] != 0

    # Per-row score lift can never exceed the weight (normalized prior in [0, 1]).
    for r in boosted:
        lift = r["fusion_score"] - base_scores[r["id"]]
        assert -1e-9 <= lift <= weight + 1e-9


def test_doc_absent_from_doc_to_oa_is_neutral() -> None:
    g = _graph()
    del g.doc_to_oa["docD"]  # docD now unmapped -> neutral prior, no crash
    out = reciprocal_rank_fusion(
        _dense(), _sparse(), citation_prior_weight=0.02, citation_graph=g
    )
    ids = {r["id"] for r in out}
    assert "docD:0" in ids
    for r in out:
        assert r["fusion_score"] == r["fusion_score"]  # no NaN


def test_missing_year_is_neutral_recency() -> None:
    dense = [
        {"id": "docA:0", "doc_id": "docA", "score": 0.9},  # no year key
        {"id": "docB:0", "doc_id": "docB", "score": 0.8, "year": None},
    ]
    sparse = [
        {"id": "docB:0", "doc_id": "docB", "score": 5.0, "year": None},
    ]
    out = reciprocal_rank_fusion(
        dense, sparse, citation_prior_weight=0.02, citation_graph=_graph()
    )
    assert {r["id"] for r in out} == {"docA:0", "docB:0"}
    for r in out:
        assert r["fusion_score"] == r["fusion_score"]


def test_search_raw_default_is_byte_identical(monkeypatch) -> None:
    """Thread-through: search_raw at the default weight must not alter ordering."""
    from graph_layout_rag.query import search as search_mod

    captured: dict = {}

    def fake_retrieve(query, **kwargs):  # noqa: ANN001, ANN202
        captured["citation_prior_weight"] = kwargs.get("citation_prior_weight")
        return []

    monkeypatch.setattr(search_mod, "retrieve_candidates", fake_retrieve)
    monkeypatch.setattr(search_mod, "diversify_candidates", lambda c, **k: c)
    monkeypatch.setattr(search_mod, "rerank_candidates", lambda q, c, **k: c)
    monkeypatch.setattr(search_mod, "format_results", lambda r, **k: r)

    search_mod.search_raw("layered graph")
    assert captured["citation_prior_weight"] == 0.0
