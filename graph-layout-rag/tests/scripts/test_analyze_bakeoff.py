"""Unit tests for scripts/analyze_bakeoff.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Make sure the scripts directory is importable
_SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from analyze_bakeoff import (
    judge_audit,
    paired_bootstrap_ci,
    rank_strategies,
)


# ---------------------------------------------------------------------------
# 1. paired_bootstrap_ci — seeded numpy fixture
# ---------------------------------------------------------------------------

class TestPairedBootstrapCI:
    """Tests for paired_bootstrap_ci."""

    def _make_run(self, scores: dict[str, float]) -> dict:
        return {"per_case_ndcg": scores}

    def test_known_positive_delta(self) -> None:
        """A always beats B by exactly 0.2 → CI should contain ~0.2, not 0."""
        rng = np.random.default_rng(0)
        case_ids = [f"c{i}" for i in range(30)]
        scores_a = {cid: float(rng.uniform(0.6, 1.0)) for cid in case_ids}
        scores_b = {cid: max(0.0, scores_a[cid] - 0.2) for cid in case_ids}

        result = paired_bootstrap_ci(
            self._make_run(scores_a),
            self._make_run(scores_b),
            n_resamples=2000,
        )

        expected_delta = np.mean(
            [scores_a[c] - scores_b[c] for c in case_ids]
        )
        assert abs(result["delta_mean"] - expected_delta) < 0.01, (
            f"delta_mean {result['delta_mean']} far from expected {expected_delta}"
        )
        # CI should not contain zero when A consistently beats B by 0.2
        assert result["contains_zero"] is False, (
            f"Expected CI to exclude zero for clear winner; got {result}"
        )
        assert result["label"] != "indistinguishable"

    def test_zero_delta_is_indistinguishable(self) -> None:
        """Identical runs → delta == 0.0, CI contains zero."""
        case_ids = [f"c{i}" for i in range(20)]
        scores = {cid: 0.7 for cid in case_ids}

        result = paired_bootstrap_ci(
            self._make_run(scores),
            self._make_run(scores),
            n_resamples=1000,
        )

        assert result["delta_mean"] == pytest.approx(0.0, abs=1e-9)
        assert result["contains_zero"] is True
        assert result["label"] == "indistinguishable"

    def test_no_common_cases_returns_nan(self) -> None:
        run_a = self._make_run({"case-a": 0.8})
        run_b = self._make_run({"case-b": 0.6})

        result = paired_bootstrap_ci(run_a, run_b)

        assert result["contains_zero"] is True
        assert "indistinguishable" in result["label"]
        import math
        assert math.isnan(result["delta_mean"])

    def test_ci_width_scales_with_variance(self) -> None:
        """Higher variance in deltas → wider CI."""
        rng = np.random.default_rng(7)
        case_ids = [f"c{i}" for i in range(40)]
        base = {cid: 0.5 for cid in case_ids}

        low_var_b = {cid: 0.45 for cid in case_ids}
        high_var_b = {cid: float(rng.uniform(0.0, 1.0)) for cid in case_ids}

        ci_low_var = paired_bootstrap_ci(
            self._make_run(base), self._make_run(low_var_b), n_resamples=2000
        )
        ci_high_var = paired_bootstrap_ci(
            self._make_run(base), self._make_run(high_var_b), n_resamples=2000
        )

        width_low = ci_low_var["ci_high"] - ci_low_var["ci_low"]
        width_high = ci_high_var["ci_high"] - ci_high_var["ci_low"]
        assert width_high > width_low, (
            f"High-variance CI ({width_high:.4f}) should be wider than "
            f"low-variance CI ({width_low:.4f})"
        )


# ---------------------------------------------------------------------------
# 2. rank_strategies — ordering
# ---------------------------------------------------------------------------

class TestRankStrategies:
    """Tests for rank_strategies."""

    def _make_runs(self) -> list[dict]:
        """Three strategies with known nDCG@10 values."""
        return [
            {
                "strategy": "hybrid",
                "qrels_path": "data/eval/qrels/catalog/qrels.json",
                "ndcg10": 0.72,
                "recall10": 0.85,
                "per_case_ndcg": {"c1": 0.8, "c2": 0.64},
            },
            {
                "strategy": "bm25",
                "qrels_path": "data/eval/qrels/catalog/qrels.json",
                "ndcg10": 0.65,
                "recall10": 0.78,
                "per_case_ndcg": {"c1": 0.7, "c2": 0.6},
            },
            {
                "strategy": "dense",
                "qrels_path": "data/eval/qrels/catalog/qrels.json",
                "ndcg10": 0.68,
                "recall10": 0.80,
                "per_case_ndcg": {"c1": 0.75, "c2": 0.61},
            },
            # This run has a different qrels path and should be excluded
            {
                "strategy": "dense",
                "qrels_path": "data/eval/qrels/pdf-deep-read/qrels.json",
                "ndcg10": 0.90,
                "recall10": 0.95,
                "per_case_ndcg": {"c1": 0.9, "c2": 0.9},
            },
        ]

    def test_rank_order_descending_ndcg(self) -> None:
        runs = self._make_runs()
        ranked = rank_strategies(runs, "catalog")

        assert len(ranked) == 3
        strategies_in_order = [r["strategy"] for r in ranked]
        assert strategies_in_order == ["hybrid", "dense", "bm25"], (
            f"Expected ['hybrid', 'dense', 'bm25'], got {strategies_in_order}"
        )

    def test_qrels_filter_excludes_other_track(self) -> None:
        runs = self._make_runs()
        ranked = rank_strategies(runs, "pdf-deep-read")
        # Only the pdf-deep-read dense run matches
        assert len(ranked) == 1
        assert ranked[0]["strategy"] == "dense"
        assert ranked[0]["mean_ndcg10"] == pytest.approx(0.90)

    def test_empty_filter_includes_all(self) -> None:
        """Empty string filter includes all runs (merges both dense runs)."""
        runs = self._make_runs()
        ranked = rank_strategies(runs, "")
        strategy_names = {r["strategy"] for r in ranked}
        assert "hybrid" in strategy_names
        assert "bm25" in strategy_names
        assert "dense" in strategy_names

    def test_ndcg_values_match(self) -> None:
        runs = self._make_runs()
        ranked = rank_strategies(runs, "catalog")
        by_name = {r["strategy"]: r for r in ranked}
        assert by_name["hybrid"]["mean_ndcg10"] == pytest.approx(0.72)
        assert by_name["bm25"]["mean_ndcg10"] == pytest.approx(0.65)
        assert by_name["dense"]["mean_ndcg10"] == pytest.approx(0.68)

    def test_no_matching_runs_returns_empty(self) -> None:
        runs = self._make_runs()
        ranked = rank_strategies(runs, "no-such-qrels")
        assert ranked == []


# ---------------------------------------------------------------------------
# 3 & 4. judge_audit — PASS and WARN
# ---------------------------------------------------------------------------

class TestJudgeAudit:
    """Tests for judge_audit using mocked pool + qrels JSON files."""

    def _write_json(self, path: Path, data: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def _make_pool(self, tmp_path: Path, cases: dict) -> Path:
        pool_path = tmp_path / "pool.json"
        self._write_json(
            pool_path,
            {
                "version": 1,
                "generated_at": "2026-01-01T00:00:00Z",
                "track": "catalog",
                "depth": 20,
                "embed_profile": "test",
                "systems": ["bm25", "dense"],
                "case_count": len(cases),
                "cases": cases,
            },
        )
        return pool_path

    def _make_qrels(self, tmp_path: Path, cases: dict) -> Path:
        qrels_path = tmp_path / "qrels.json"
        self._write_json(
            qrels_path,
            {
                "version": 1,
                "generated_at": "2026-01-01T00:00:00Z",
                "judge_model": "test-model",
                "relevance_threshold": 2,
                "track": "catalog",
                "pool": str(tmp_path / "pool.json"),
                "judge_agreement": {},
                "cases": cases,
            },
        )
        return qrels_path

    def _write_seeds(self, tmp_path: Path, case_ids: list[str]) -> Path:
        seeds_path = tmp_path / "seeds.json"
        self._write_json(seeds_path, case_ids)
        return seeds_path

    def test_pass_gate(self, tmp_path: Path) -> None:
        """Dense and BM25 docs have similar judge scores → PASS."""
        pool_cases = {}
        qrels_cases = {}
        seed_ids = []

        # 10 explicitly named seed cases, each with one curated doc (grade=3)
        # and one non-curated doc (grade=0), giving variance for Spearman.
        for i in range(10):
            case_id = f"seed-case-{i}"
            seed_ids.append(case_id)
            curated_doc = f"curated-doc-{i}"
            noise_doc = f"noise-doc-{i}"
            pool_cases[case_id] = {
                "query": f"query {i}",
                "category": "layer-assignment",
                "pdf_only": False,
                "curated_relevant": [curated_doc],
                "pooled": {
                    curated_doc: {
                        "canonical_doc_id": curated_doc,
                        "doc_id": curated_doc,
                        "title": f"Curated Paper {i}",
                        "excerpt": "...",
                        "systems": ["bm25", "dense"],
                        "curated": True,
                        "doc_kind": "metadata",
                    },
                    noise_doc: {
                        "canonical_doc_id": noise_doc,
                        "doc_id": noise_doc,
                        "title": f"Noise Paper {i}",
                        "excerpt": "...",
                        "systems": ["bm25", "dense"],
                        "curated": False,
                        "doc_kind": "metadata",
                    },
                },
            }
            qrels_cases[case_id] = {
                curated_doc: {
                    "grade": 3,
                    "reason": "highly relevant",
                    "systems": ["bm25", "dense"],
                    "curated": True,
                },
                noise_doc: {
                    "grade": 0,
                    "reason": "not relevant",
                    "systems": ["bm25", "dense"],
                    "curated": False,
                },
            }

        # Non-seed cases with similar dense and BM25 doc scores (no bias)
        for i in range(10, 30):
            case_id = f"nonseed-case-{i}"
            dense_doc = f"dense-doc-{i}"
            bm25_doc = f"bm25-doc-{i}"
            pool_cases[case_id] = {
                "query": f"query {i}",
                "category": "crossing",
                "pdf_only": False,
                "curated_relevant": [],
                "pooled": {
                    dense_doc: {
                        "canonical_doc_id": dense_doc,
                        "doc_id": dense_doc,
                        "title": f"Dense Paper {i}",
                        "excerpt": "...",
                        "systems": ["dense", "colbert"],
                        "curated": False,
                        "doc_kind": "metadata",
                    },
                    bm25_doc: {
                        "canonical_doc_id": bm25_doc,
                        "doc_id": bm25_doc,
                        "title": f"BM25 Paper {i}",
                        "excerpt": "...",
                        "systems": ["bm25"],
                        "curated": False,
                        "doc_kind": "metadata",
                    },
                },
            }
            # Both dense and BM25 docs grade similarly (no bias)
            qrels_cases[case_id] = {
                dense_doc: {
                    "grade": 2,
                    "reason": "relevant",
                    "systems": ["dense", "colbert"],
                    "curated": False,
                },
                bm25_doc: {
                    "grade": 2,
                    "reason": "relevant",
                    "systems": ["bm25"],
                    "curated": False,
                },
            }

        pool_path = self._make_pool(tmp_path, pool_cases)
        qrels_path = self._make_qrels(tmp_path, qrels_cases)
        seeds_path = self._write_seeds(tmp_path, seed_ids)

        result = judge_audit(str(pool_path), str(qrels_path), str(seeds_path))

        assert result["gate"] == "PASS", f"Expected PASS, got: {result}"
        assert result["bias_flag"] is False
        assert result["seed_spearman"] >= 0.70

    def test_warn_gate_dense_bias(self, tmp_path: Path) -> None:
        """Judge disagrees with curated truth by provenance → WARN.

        NOTE (corrected semantics): the judge is source-blind, so a higher
        *relevance rate* among dense-only docs is NOT bias — it is the de-biased
        pool's payoff. Genuine judge bias only shows up as the judge disagreeing
        with GROUND TRUTH differently by provenance. Here curated (truly relevant)
        dense-only docs are wrongly graded 0 while curated bm25-only docs are
        correctly graded 3, so the disagreement gap trips the gate.
        """
        pool_cases = {}
        qrels_cases = {}
        seed_ids = []

        # Seed cases that agree well (high kappa/spearman): curated=3, noise=0
        for i in range(10):
            case_id = f"seed-case-{i}"
            seed_ids.append(case_id)
            curated_doc = f"curated-doc-{i}"
            noise_doc = f"noise-doc-{i}"
            pool_cases[case_id] = {
                "query": f"query {i}",
                "category": "layer-assignment",
                "pdf_only": False,
                "curated_relevant": [curated_doc],
                "pooled": {
                    curated_doc: {
                        "canonical_doc_id": curated_doc,
                        "doc_id": curated_doc,
                        "title": f"Curated Paper {i}",
                        "excerpt": "...",
                        "systems": ["bm25", "dense"],
                        "curated": True,
                        "doc_kind": "metadata",
                    },
                    noise_doc: {
                        "canonical_doc_id": noise_doc,
                        "doc_id": noise_doc,
                        "title": f"Noise Paper {i}",
                        "excerpt": "...",
                        "systems": ["bm25", "dense"],
                        "curated": False,
                        "doc_kind": "metadata",
                    },
                },
            }
            qrels_cases[case_id] = {
                curated_doc: {
                    "grade": 3,
                    "reason": "highly relevant",
                    "systems": ["bm25", "dense"],
                    "curated": True,
                },
                noise_doc: {
                    "grade": 0,
                    "reason": "not relevant",
                    "systems": ["bm25", "dense"],
                    "curated": False,
                },
            }

        # Curated dense-only docs (truth=relevant) wrongly judged 0 → disagreement.
        for i in range(10, 30):
            case_id = f"dense-truth-case-{i}"
            doc = f"dense-truth-doc-{i}"
            pool_cases[case_id] = {
                "query": f"query {i}",
                "category": "crossing",
                "pdf_only": False,
                "curated_relevant": [doc],
                "pooled": {
                    doc: {
                        "canonical_doc_id": doc,
                        "doc_id": doc,
                        "title": f"Dense Paper {i}",
                        "excerpt": "...",
                        "systems": ["dense", "colbert"],
                        "curated": True,
                        "doc_kind": "metadata",
                    },
                },
            }
            qrels_cases[case_id] = {
                doc: {
                    "grade": 0,
                    "reason": "judge wrongly rejects dense",
                    "systems": ["dense", "colbert"],
                    "curated": True,
                },
            }

        # Curated bm25-only docs (truth=relevant) correctly judged 3 → no disagreement.
        for i in range(30, 50):
            case_id = f"bm25-truth-case-{i}"
            doc = f"bm25-truth-doc-{i}"
            pool_cases[case_id] = {
                "query": f"query {i}",
                "category": "crossing",
                "pdf_only": False,
                "curated_relevant": [doc],
                "pooled": {
                    doc: {
                        "canonical_doc_id": doc,
                        "doc_id": doc,
                        "title": f"BM25 Paper {i}",
                        "excerpt": "...",
                        "systems": ["bm25"],
                        "curated": True,
                        "doc_kind": "metadata",
                    },
                },
            }
            qrels_cases[case_id] = {
                doc: {
                    "grade": 3,
                    "reason": "judge accepts bm25",
                    "systems": ["bm25"],
                    "curated": True,
                },
            }

        pool_path = self._make_pool(tmp_path, pool_cases)
        qrels_path = self._make_qrels(tmp_path, qrels_cases)
        seeds_path = self._write_seeds(tmp_path, seed_ids)

        result = judge_audit(str(pool_path), str(qrels_path), str(seeds_path))

        assert result["gate"] == "WARN", f"Expected WARN, got: {result}"
        assert result["bias_flag"] is True
        assert result["dense_disagree_rate"] > result["bm25_disagree_rate"] + 0.25

    def test_fail_gate_poor_seed_agreement(self, tmp_path: Path) -> None:
        """Judge scores disagree with curated grades on seed cases → FAIL."""
        pool_cases = {}
        qrels_cases = {}
        seed_ids = []

        # Seed cases: curated doc graded 0 (judge disagrees) + noise doc graded 3
        # → inverted relationship, very negative Spearman and low kappa.
        for i in range(15):
            case_id = f"seed-case-{i}"
            seed_ids.append(case_id)
            curated_doc = f"curated-doc-{i}"
            noise_doc = f"noise-doc-{i}"
            pool_cases[case_id] = {
                "query": f"query {i}",
                "category": "layer-assignment",
                "pdf_only": False,
                "curated_relevant": [curated_doc],
                "pooled": {
                    curated_doc: {
                        "canonical_doc_id": curated_doc,
                        "doc_id": curated_doc,
                        "title": f"Curated Paper {i}",
                        "excerpt": "...",
                        "systems": ["bm25", "dense"],
                        "curated": True,
                        "doc_kind": "metadata",
                    },
                    noise_doc: {
                        "canonical_doc_id": noise_doc,
                        "doc_id": noise_doc,
                        "title": f"Noise Paper {i}",
                        "excerpt": "...",
                        "systems": ["bm25", "dense"],
                        "curated": False,
                        "doc_kind": "metadata",
                    },
                },
            }
            # Judge inverts: grade=0 for curated, grade=3 for noise → poor agreement
            qrels_cases[case_id] = {
                curated_doc: {
                    "grade": 0,
                    "reason": "judge disagrees",
                    "systems": ["bm25", "dense"],
                    "curated": True,
                },
                noise_doc: {
                    "grade": 3,
                    "reason": "judge over-rates noise",
                    "systems": ["bm25", "dense"],
                    "curated": False,
                },
            }

        pool_path = self._make_pool(tmp_path, pool_cases)
        qrels_path = self._make_qrels(tmp_path, qrels_cases)
        seeds_path = self._write_seeds(tmp_path, seed_ids)

        result = judge_audit(str(pool_path), str(qrels_path), str(seeds_path))

        assert result["gate"] == "FAIL", f"Expected FAIL, got: {result}"
        assert result["seed_spearman"] < 0.70 or result["seed_kappa"] < 0.70
