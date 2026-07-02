from __future__ import annotations

import pytest
from rag_common.evals import ablation
from rag_common.evals.concentration import ConcentrationReport


def _report(z: float | None) -> ConcentrationReport:
    return ConcentrationReport(surveys=[], mean_score=0.5, mean_z_uniform=z, mean_z_size_matched=z)


def _arm(name: str, alpha: float, z: float | None, **kw) -> ablation.AblationArm:
    return ablation.AblationArm(
        name=name, alpha=alpha, cluster_count=3, concentration=_report(z), **kw
    )


def _pc(mean_diff: float | None, excludes_zero: bool | None, **kw) -> ablation.PairedComparison:
    return ablation.PairedComparison(
        against="x",
        n_surveys=kw.pop("n_surveys", 10),
        mean_diff=mean_diff,
        ci_low=kw.pop("ci_low", None),
        ci_high=kw.pop("ci_high", None),
        sign_test_p=kw.pop("sign_test_p", None),
        excludes_zero=excludes_zero,
    )


# --- promotion gate (paired-CI based) ---------------------------------------------------


def test_gate_promotes_when_hybrid_beats_both_arms_ci_excludes_zero():
    arms = {
        "hybrid": _arm("hybrid", 0.5, 2.0),
        "text_only": _arm("text_only", 0.0, 1.0),
        "citation_only": _arm("citation_only", 1.0, 0.5),
    }
    paired = {
        "text_only": _pc(0.1, True, ci_low=0.02, ci_high=0.18),
        "citation_only": _pc(0.2, True, ci_low=0.05, ci_high=0.35),
    }
    result = ablation.evaluate_promotion_gate(arms, paired)
    assert result.verdict == "PROMOTE"
    assert result.passed is True


def test_gate_blocks_when_hybrid_loses_to_one_arm():
    arms = {
        "hybrid": _arm("hybrid", 0.5, 1.0),
        "text_only": _arm("text_only", 0.0, 1.5),
        "citation_only": _arm("citation_only", 1.0, 0.5),
    }
    paired = {
        "text_only": _pc(-0.1, True, ci_low=-0.18, ci_high=-0.02),
        "citation_only": _pc(0.2, True, ci_low=0.05, ci_high=0.35),
    }
    result = ablation.evaluate_promotion_gate(arms, paired)
    assert result.verdict == "BLOCK"
    assert result.passed is False


def test_gate_ties_when_cis_include_zero_and_no_loss():
    arms = {
        "hybrid": _arm("hybrid", 0.5, 1.0),
        "text_only": _arm("text_only", 0.0, 1.0),
        "citation_only": _arm("citation_only", 1.0, 0.5),
    }
    paired = {
        "text_only": _pc(0.02, False, ci_low=-0.05, ci_high=0.09),
        "citation_only": _pc(0.1, True, ci_low=0.01, ci_high=0.2),
    }
    result = ablation.evaluate_promotion_gate(arms, paired)
    assert result.verdict == "TIE"
    assert result.passed is False


def test_gate_blocks_when_hybrid_missing():
    arms = {
        "text_only": _arm("text_only", 0.0, 1.0),
        "citation_only": _arm("citation_only", 1.0, 0.5),
    }
    result = ablation.evaluate_promotion_gate(arms, {})
    assert result.verdict == "BLOCK"


def test_gate_blocks_when_an_arm_is_missing_entirely():
    arms = {"hybrid": _arm("hybrid", 0.5, 2.0), "text_only": _arm("text_only", 0.0, 1.0)}
    result = ablation.evaluate_promotion_gate(
        arms, {"text_only": _pc(0.1, True, ci_low=0.01, ci_high=0.2)}
    )
    assert result.verdict == "BLOCK"
    assert any("citation_only" in r for r in result.reasons)


def test_gate_blocks_when_paired_data_missing_for_an_arm():
    arms = {
        "hybrid": _arm("hybrid", 0.5, 2.0),
        "text_only": _arm("text_only", 0.0, 1.0),
        "citation_only": _arm("citation_only", 1.0, 0.5),
    }
    paired = {"text_only": _pc(0.1, True, ci_low=0.01, ci_high=0.2)}  # citation_only missing
    result = ablation.evaluate_promotion_gate(arms, paired)
    assert result.verdict == "BLOCK"
    assert any("citation_only" in r and "no paired" in r for r in result.reasons)


# --- sign test / bootstrap CI ------------------------------------------------------------


def test_sign_test_hand_computed():
    # 8 positive, 2 negative out of 10 -> two-sided exact binomial p
    diffs = [1, 1, 1, 1, 1, 1, 1, 1, -1, -1]
    result = ablation.sign_test(diffs)
    assert result["n"] == 10
    assert result["positive"] == 8
    assert result["negative"] == 2
    # P(k<=2 or k>=8 | n=10,p=.5) = 2 * sum_{i=0}^{2} C(10,i) / 2^10
    expected = 2 * (1 + 10 + 45) / 1024
    assert result["p_value"] == pytest.approx(expected)


def test_sign_test_all_zero_diffs_returns_none_pvalue():
    result = ablation.sign_test([0.0, 0.0, 0.0])
    assert result["n"] == 0
    assert result["p_value"] is None


def test_sign_test_ties_excluded_from_n():
    result = ablation.sign_test([1.0, -1.0, 0.0, 0.0])
    assert result["n"] == 2


def test_bootstrap_ci_constant_diffs_has_zero_width_ci():
    diffs = [0.5] * 20
    result = ablation.bootstrap_ci(diffs, n_boot=200, seed=1)
    assert result["mean"] == pytest.approx(0.5)
    assert result["ci_low"] == pytest.approx(0.5)
    assert result["ci_high"] == pytest.approx(0.5)


def test_bootstrap_ci_empty_returns_none():
    result = ablation.bootstrap_ci([], n_boot=100)
    assert result["mean"] is None
    assert result["ci_low"] is None


def test_bootstrap_ci_excludes_zero_when_all_positive():
    diffs = [0.1, 0.12, 0.09, 0.11, 0.15, 0.08, 0.13, 0.1] * 3
    result = ablation.bootstrap_ci(diffs, n_boot=1000, seed=2)
    assert result["ci_low"] > 0


def test_bootstrap_ci_includes_zero_when_diffs_straddle_zero():
    diffs = [0.5, -0.5, 0.4, -0.4, 0.1, -0.1, 0.3, -0.3]
    result = ablation.bootstrap_ci(diffs, n_boot=1000, seed=3)
    assert result["ci_low"] < 0 < result["ci_high"]


# --- common-support scoring ---------------------------------------------------------------


def test_restrict_to_common_support_drops_refs_not_covered_by_all_arms():
    # doc a/b in a real 2-doc cluster under hybrid; under citation_only, "b" is a
    # singleton (its own 1-member cluster) -> not covered there.
    gold = {"survey1": {"_all": ["a", "b", "c"]}}
    partitions = {
        "hybrid": {"a": 0, "b": 0, "c": 1},
        "citation_only": {"a": 0, "b": 99, "c": 1},  # b is a singleton cluster (id 99)
    }
    # "c" also needs to be in a >=2-doc cluster somewhere to survive; make cluster 1 size-1
    # under citation_only too, so only "a" is common-support -- with <2 refs the whole
    # survey should be dropped.
    result = ablation.restrict_to_common_support(gold, partitions)
    assert result == {}


def test_restrict_to_common_support_keeps_refs_covered_everywhere():
    gold = {"survey1": {"_all": ["a", "b", "c", "d"]}}
    partitions = {
        "hybrid": {"a": 0, "b": 0, "c": 1, "d": 1},
        "citation_only": {"a": 5, "b": 5, "c": 6, "d": 6},
    }
    result = ablation.restrict_to_common_support(gold, partitions)
    assert set(result["survey1"]["_all"]) == {"a", "b", "c", "d"}


def test_restrict_to_common_support_drops_survey_with_fewer_than_two_common_refs():
    gold = {"survey1": {"_all": ["a", "b"]}}
    partitions = {
        "hybrid": {"a": 0, "b": 0},
        "citation_only": {"a": 5, "b": 6},  # both singletons under citation_only
    }
    result = ablation.restrict_to_common_support(gold, partitions)
    assert result == {}


# --- resolution sweep picks matched K on synthetic graphs --------------------------------

igraph = pytest.importorskip("igraph")
pytest.importorskip("leidenalg")


def _grid_doc_vectors(n_clusters: int, per_cluster: int) -> dict[str, list[float]]:
    """`n_clusters` well-separated 2D blobs (one-hot-ish), `per_cluster` docs each — a
    resolution sweep should be able to hit any target K from 1 up to n_clusters on this."""
    import math

    vectors: dict[str, list[float]] = {}
    for c in range(n_clusters):
        angle = 2 * math.pi * c / n_clusters
        base = [math.cos(angle), math.sin(angle)]
        for i in range(per_cluster):
            jitter = 0.01 * (i - per_cluster / 2)
            vectors[f"c{c}d{i}"] = [base[0] + jitter, base[1] - jitter]
    return vectors


def test_sweep_resolution_hits_low_target_k_on_fine_grained_graph():
    # 12 tight clusters -> at resolution=1.0, Leiden should find close to 12; sweeping to a
    # much lower target_k (e.g. 2) must find a resolution that merges them down.
    doc_vectors = _grid_doc_vectors(n_clusters=12, per_cluster=4)
    gold: dict[str, dict[str, list[str]]] = {}
    partition, points = ablation.sweep_resolution(
        doc_vectors, None, gold, alpha=0.0, target_k=2, top_k=5, seed=1
    )
    achieved_k = ablation.cluster_count_min2(partition)
    # Best achievable point in the swept grid should be closer to 2 than the natural
    # resolution=1.0 point (which sits near 12).
    natural = next(p for p in points if p.resolution == 1.0)
    assert abs(achieved_k - 2) <= abs(natural.cluster_count_min2 - 2)


def test_sweep_resolution_curve_covers_every_grid_point():
    doc_vectors = _grid_doc_vectors(n_clusters=6, per_cluster=3)
    _partition, points = ablation.sweep_resolution(
        doc_vectors, None, {}, alpha=0.0, target_k=3, top_k=5, seed=1
    )
    assert [p.resolution for p in points] == list(ablation.DEFAULT_RESOLUTION_GRID)


# --- coverage stats -------------------------------------------------------------------


def test_cluster_coverage_all_singletons_is_zero_coverage():
    from rag_common.evals.concentration import cluster_coverage

    partition = {"a": 0, "b": 1, "c": 2}
    stats = cluster_coverage(partition, ["a", "b", "c"])
    assert stats.coverage == 0.0
    assert stats.singleton_share == 1.0


def test_cluster_coverage_fully_clustered_is_one():
    from rag_common.evals.concentration import cluster_coverage

    partition = {"a": 0, "b": 0, "c": 1, "d": 1}
    stats = cluster_coverage(partition, ["a", "b", "c", "d"])
    assert stats.coverage == 1.0
    assert stats.singleton_share == 0.0


def test_cluster_coverage_counts_missing_docs_in_denominator_only():
    from rag_common.evals.concentration import cluster_coverage

    partition = {"a": 0, "b": 0}
    stats = cluster_coverage(partition, ["a", "b", "missing"])
    assert stats.doc_count == 3
    assert stats.covered_count == 2
    assert stats.coverage == pytest.approx(2 / 3)


def test_cluster_count_min2():
    from rag_common.evals.concentration import cluster_count_min2

    partition = {"a": 0, "b": 0, "c": 1, "d": 2, "e": 2}
    assert cluster_count_min2(partition) == 2  # cluster 0 (2 docs) and 2 (2 docs); 1 is a singleton


# --- end-to-end ---------------------------------------------------------------------------


def test_run_ablation_end_to_end_no_citation_signal():
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


def test_run_ablation_reports_sweep_curves_and_target_k():
    doc_vectors = _grid_doc_vectors(n_clusters=6, per_cluster=3)
    gold = {f"survey{i}": {"_all": [f"c{i}d0", f"c{i}d1", f"c{i}d2"]} for i in range(6)}
    payload = ablation.run_ablation(doc_vectors, None, gold, top_k=5, n_random=5)
    assert payload["target_k"] is not None
    assert "text_only" in payload["sweep_curves"]
    assert len(payload["sweep_curves"]["text_only"]) == len(ablation.DEFAULT_RESOLUTION_GRID)
