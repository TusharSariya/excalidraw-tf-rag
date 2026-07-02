from __future__ import annotations

import pytest
from rag_common.evals import ablation
from rag_common.evals.concentration import ConcentrationReport


def _report(z: float | None) -> ConcentrationReport:
    return ConcentrationReport(surveys=[], mean_score=0.5, mean_z_uniform=z, mean_z_size_matched=z)


def _arm(name: str, alpha: float, z: float | None) -> ablation.AblationArm:
    return ablation.AblationArm(name=name, alpha=alpha, cluster_count=3, concentration=_report(z))


def test_gate_promotes_when_hybrid_beats_both_singles():
    arms = {
        "hybrid": _arm("hybrid", 0.5, 2.0),
        "text_only": _arm("text_only", 0.0, 1.0),
        "citation_only": _arm("citation_only", 1.0, 0.5),
    }
    result = ablation.evaluate_promotion_gate(arms)
    assert result.passed is True


def test_gate_blocks_when_hybrid_loses_to_text_only():
    arms = {
        "hybrid": _arm("hybrid", 0.5, 1.0),
        "text_only": _arm("text_only", 0.0, 1.5),
        "citation_only": _arm("citation_only", 1.0, 0.5),
    }
    result = ablation.evaluate_promotion_gate(arms)
    assert result.passed is False


def test_gate_blocks_when_hybrid_loses_to_citation_only():
    arms = {
        "hybrid": _arm("hybrid", 0.5, 1.0),
        "text_only": _arm("text_only", 0.0, 0.5),
        "citation_only": _arm("citation_only", 1.0, 1.5),
    }
    result = ablation.evaluate_promotion_gate(arms)
    assert result.passed is False


def test_gate_blocks_when_hybrid_missing():
    arms = {
        "text_only": _arm("text_only", 0.0, 1.0),
        "citation_only": _arm("citation_only", 1.0, 0.5),
    }
    result = ablation.evaluate_promotion_gate(arms)
    assert result.passed is False


def test_gate_blocks_when_a_single_signal_arm_missing_zscore():
    arms = {
        "hybrid": _arm("hybrid", 0.5, 2.0),
        "text_only": _arm("text_only", 0.0, None),
        "citation_only": _arm("citation_only", 1.0, 0.5),
    }
    result = ablation.evaluate_promotion_gate(arms)
    assert result.passed is False
    assert any("text_only" in r for r in result.reasons)


def test_gate_ties_do_not_pass():
    arms = {
        "hybrid": _arm("hybrid", 0.5, 1.0),
        "text_only": _arm("text_only", 0.0, 1.0),
        "citation_only": _arm("citation_only", 1.0, 0.5),
    }
    result = ablation.evaluate_promotion_gate(arms)
    assert result.passed is False


igraph = pytest.importorskip("igraph")
pytest.importorskip("leidenalg")


def test_run_ablation_end_to_end_no_citation_signal():
    # Two well-separated text clusters, no citation adjacency -> only text_only arm runs,
    # citation_only/hybrid are skipped, gate blocks with a clear reason (nothing to compare).
    doc_vectors = {
        "a1": [1.0, 0.0],
        "a2": [0.99, 0.01],
        "b1": [0.0, 1.0],
        "b2": [0.01, 0.99],
    }
    gold = {"survey1": {"_all": ["a1", "a2"]}}
    payload = ablation.run_ablation(doc_vectors, None, gold, top_k=3, n_random=10)
    assert payload["citation_signal_available"] is False
    assert "text_only" in payload["arms"]
    assert "hybrid" not in payload["arms"]
    assert "citation_only" not in payload["arms"]
    assert payload["gate"]["gate"] == "BLOCK"

    md = ablation.render_markdown_report(payload)
    assert "E1 map-gate report" in md
    assert "text_only" in md
