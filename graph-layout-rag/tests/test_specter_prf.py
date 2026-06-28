"""SPECTER2 PRF re-rank strategy (T8 A/B arm).

The strategy is a *re-ranker*: it builds a pseudo-relevance-feedback centroid from
the top text-ranked docs' SPECTER2 vectors and multiplicatively blends each candidate
doc's cosine-to-centroid into the text fusion score. These tests mock retrieval and the
SPECTER2 store so the blend math is validated without an index or GPU.
"""
from __future__ import annotations

import math

import graph_layout_rag.eval.strategies as strat
from graph_layout_rag.eval.gold_cases import EvalCase
from graph_layout_rag.eval.strategies import SpecterPRFStrategy, _dot, _mean_unit


def _case() -> EvalCase:
    return EvalCase(id="c1", query="layered graph drawing", relevant_doc_ids=frozenset({"d1"}))


# ----------------------------------------------------------------- math helpers
def test_dot_is_cosine_for_unit_vectors():
    assert _dot([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert _dot([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_mean_unit_is_unit_length():
    out = _mean_unit([[1.0, 0.0], [0.0, 1.0]])
    assert math.isclose(math.sqrt(out[0] ** 2 + out[1] ** 2), 1.0, rel_tol=1e-9)
    assert math.isclose(out[0], out[1], rel_tol=1e-9)  # symmetric seeds → diagonal


# ----------------------------------------------------------------- blend behavior
def _patch(monkeypatch, *, candidates, vectors, db_exists=True):
    """Wire retrieve_candidates/format_results/store so the strategy runs offline."""
    monkeypatch.setattr(strat, "_filters", lambda *a, **k: None)

    import graph_layout_rag.query.retrieve as retrieve
    import graph_layout_rag.query.search as search
    import graph_layout_rag.citation_store as store
    import graph_layout_rag.paths as paths

    monkeypatch.setattr(retrieve, "retrieve_candidates", lambda *a, **k: [dict(c) for c in candidates])
    # format_results: return the (already re-sorted) rows' doc_ids in order.
    monkeypatch.setattr(
        search, "format_results", lambda rows, **k: [{"doc_id": r["doc_id"], "score": r.get("fusion_score")} for r in rows]
    )
    monkeypatch.setattr(store, "connect", lambda *a, **k: object())
    monkeypatch.setattr(store, "specter2_for_doc", lambda db, doc: vectors.get(doc))

    class _P:
        @staticmethod
        def exists() -> bool:
            return db_exists

    monkeypatch.setattr(paths, "CITATIONS_DB_PATH", _P)
    # The strategy closes db; our stub object has no close — give it one.
    monkeypatch.setattr(store, "connect", lambda *a, **k: type("C", (), {"close": lambda self: None})())


def test_specter_boost_promotes_related_doc(monkeypatch):
    # d_low has the higher text score but is unrelated; d_hi is SPECTER-aligned with the
    # seed centroid and should overtake it once the boost applies.
    candidates = [
        {"doc_id": "seed", "fusion_score": 1.0},
        {"doc_id": "d_low", "fusion_score": 0.90},
        {"doc_id": "d_hi", "fusion_score": 0.80},
    ]
    vectors = {
        "seed": [1.0, 0.0],
        "d_hi": [1.0, 0.0],   # cosine 1.0 with centroid
        "d_low": [0.0, 1.0],  # cosine 0.0
    }
    _patch(monkeypatch, candidates=candidates, vectors=vectors)
    s = SpecterPRFStrategy(seed_docs=1, alpha=0.5)
    out = s.run(_case(), embed_profile="x", top=10)
    order = [r["doc_id"] for r in out]
    # seed stays #1 (1.0*1.5=1.5); d_hi 0.80*1.5=1.20 overtakes d_low 0.90*1.0=0.90.
    assert order == ["seed", "d_hi", "d_low"]


def test_alpha_zero_is_passthrough(monkeypatch):
    candidates = [
        {"doc_id": "a", "fusion_score": 0.5},
        {"doc_id": "b", "fusion_score": 0.9},
    ]
    _patch(monkeypatch, candidates=candidates, vectors={"a": [1.0, 0.0], "b": [0.0, 1.0]})
    s = SpecterPRFStrategy(alpha=0.0)
    out = s.run(_case(), embed_profile="x", top=10)
    # alpha=0 short-circuits before any re-scoring: original order preserved.
    assert [r["doc_id"] for r in out] == ["a", "b"]


def test_no_specter_vectors_degrades_to_hybrid(monkeypatch):
    candidates = [{"doc_id": "a", "fusion_score": 0.5}, {"doc_id": "b", "fusion_score": 0.9}]
    _patch(monkeypatch, candidates=candidates, vectors={})  # store has nothing
    s = SpecterPRFStrategy(alpha=0.5)
    out = s.run(_case(), embed_profile="x", top=10)
    assert [r["doc_id"] for r in out] == ["a", "b"]  # untouched hybrid order


def test_missing_db_degrades_to_hybrid(monkeypatch):
    candidates = [{"doc_id": "a", "fusion_score": 0.5}]
    _patch(monkeypatch, candidates=candidates, vectors={"a": [1.0, 0.0]}, db_exists=False)
    s = SpecterPRFStrategy(alpha=0.5)
    out = s.run(_case(), embed_profile="x", top=10)
    assert [r["doc_id"] for r in out] == ["a"]


def test_registered_in_all_strategies():
    from graph_layout_rag.eval.strategies import ALL_STRATEGIES, strategy_registry

    assert "hybrid_specter_prf" in ALL_STRATEGIES
    assert "hybrid_specter_prf" in strategy_registry()
