from __future__ import annotations

import pytest
from rag_common.evals import concentration as conc


def test_herfindahl_perfect_concentration_one_cluster():
    assert conc.herfindahl([1, 1, 1, 1]) == 1.0


def test_herfindahl_uniform_spread():
    # 4 items, 4 distinct clusters -> HHI = 4 * (1/4)^2 = 0.25
    assert conc.herfindahl([1, 2, 3, 4]) == 0.25


def test_herfindahl_empty():
    assert conc.herfindahl([]) == 0.0


def test_score_refs_below_two_placeable_is_none():
    partition = {"a": 0, "b": 1}
    assert conc.score_refs(["a"], partition) is None
    assert conc.score_refs(["missing1", "missing2"], partition) is None


def test_score_refs_perfect_vs_spread():
    partition = {f"d{i}": 0 for i in range(5)} | {f"e{i}": i + 1 for i in range(5)}
    perfect = conc.score_refs([f"d{i}" for i in range(5)], partition)
    spread = conc.score_refs([f"e{i}" for i in range(5)], partition)
    assert perfect == 1.0
    assert spread == pytest.approx(0.2)
    assert perfect > spread


def test_size_matched_random_partition_preserves_size_distribution():
    partition = {f"d{i}": (0 if i < 8 else 1) for i in range(10)}
    import random

    rng = random.Random(1)
    from collections import Counter

    original_sizes = Counter(partition.values())
    for _ in range(20):
        shuffled = conc.size_matched_random_partition(partition, rng=rng)
        assert Counter(shuffled.values()) == original_sizes
        assert set(shuffled) == set(partition)


def test_uniform_random_partition_uses_only_given_cluster_ids():
    import random

    rng = random.Random(2)
    doc_ids = [f"d{i}" for i in range(20)]
    cluster_ids = [0, 1, 2]
    partition = conc.uniform_random_partition(doc_ids, cluster_ids, rng=rng)
    assert set(partition) == set(doc_ids)
    assert set(partition.values()) <= set(cluster_ids)


def test_score_survey_z_score_sanity_perfect_concentration_scores_high_z():
    # Build a partition where survey refs all sit in one cluster amid a bigger corpus
    # with many small distinct clusters, so a random reshuffle should very rarely match.
    partition: dict[str, int] = {}
    refs = [f"ref{i}" for i in range(10)]
    for r in refs:
        partition[r] = 0
    for i in range(30):
        partition[f"other{i}"] = i + 1  # 30 singleton-ish clusters

    result = conc.score_survey("survey1", {"_all": refs}, partition, n_random=50, seed=7)
    assert result is not None
    assert result.mean_score == 1.0
    assert result.size_matched_null is not None
    assert result.size_matched_null.mean < 1.0
    assert result.size_matched_null.z is not None
    assert result.size_matched_null.z > 0


def test_score_survey_no_placeable_refs_returns_none():
    partition = {"a": 0, "b": 1}
    result = conc.score_survey("survey1", {"_all": ["missing1", "missing2"]}, partition, n_random=5)
    assert result is None


def test_concentration_report_aggregates_multiple_surveys():
    partition: dict[str, int] = {}
    refs1 = [f"r1_{i}" for i in range(6)]
    refs2 = [f"r2_{i}" for i in range(6)]
    for r in refs1:
        partition[r] = 0
    for r in refs2:
        partition[r] = 1
    for i in range(20):
        partition[f"other{i}"] = i + 2

    gold = {"survey1": {"_all": refs1}, "survey2": {"_all": refs2}}
    report = conc.concentration_report(gold, partition, n_random=30, seed=3)
    assert report.mean_score is not None
    assert len(report.surveys) == 2
    d = report.to_dict()
    assert d["survey_count"] == 2
    assert d["schema_version"] == 1
