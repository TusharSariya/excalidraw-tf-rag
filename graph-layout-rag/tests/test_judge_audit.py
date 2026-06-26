"""Validity tests for the corrected ``judge_audit`` provenance-bias gate.

Background / the bug this guards against
----------------------------------------
``judge_audit`` (scripts/analyze_bakeoff.py) gates a possibly-biased LLM judge.
Its old provenance check compared the raw *relevance rate* of dense-only vs
bm25-only pooled docs and flagged "bias" when dense scored >0.25 higher. That is
a FALSE-NEGATIVE generator: the judge is source-blind (its prompt never reveals
which retrieval system surfaced a doc), so a relevant dense-only doc cannot be
judge favouritism — it is the *designed payoff* of de-biased pooling (a relevant
doc BM25 missed). Counting those as bias could wrongly WARN/FAIL a fair judge.

The corrected check conditions on GROUND TRUTH: over curated-labelled docs
(curated=True ⇒ truly relevant), it compares the judge↔truth disagreement rate
split by retrieval provenance. A fair judge disagrees with truth at the same rate
regardless of provenance (gap ~0); a judge that systematically mis-grades one
provenance's docs relative to their true relevance still produces a large gap.

Critical assertions here:
  * a clean / agreeing judge PASSES;
  * a judge that found extra relevant dense-only docs (no curated truth) does NOT
    trip the gate (the false-negative is fixed);
  * a *genuinely* provenance-biased judge — one that disagrees with curated truth
    far more for one provenance — STILL FAILS the gate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make sure the scripts directory is importable (mirrors test_analyze_bakeoff.py).
_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from analyze_bakeoff import judge_audit  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #
def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _pool(tmp: Path, cases: dict) -> Path:
    p = tmp / "pool.json"
    _write_json(
        p,
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
    return p


def _qrels(tmp: Path, cases: dict) -> Path:
    p = tmp / "qrels.json"
    _write_json(
        p,
        {
            "version": 1,
            "generated_at": "2026-01-01T00:00:00Z",
            "judge_model": "test-model",
            "relevance_threshold": 2,
            "track": "catalog",
            "pool": str(tmp / "pool.json"),
            "judge_agreement": {},
            "cases": cases,
        },
    )
    return p


def _seeds(tmp: Path, ids: list[str]) -> Path:
    p = tmp / "seeds.json"
    _write_json(p, ids)
    return p


def _seed_case(i: int, *, curated_grade: int, noise_grade: int) -> tuple[dict, dict]:
    """One seed case: a curated doc (true=relevant) + a non-curated noise doc.

    ``curated_grade``/``noise_grade`` are what the *judge* assigned, letting a
    fixture model an agreeing judge (3/0) or a disagreeing one (0/3).
    """
    cid = f"seed-case-{i}"
    cur, noise = f"curated-doc-{i}", f"noise-doc-{i}"
    pool = {
        cid: {
            "query": f"query {i}",
            "category": "layer-assignment",
            "pdf_only": False,
            "curated_relevant": [cur],
            "pooled": {
                cur: {
                    "canonical_doc_id": cur,
                    "doc_id": cur,
                    "title": f"Curated Paper {i}",
                    "systems": {"bm25": 1, "dense": 1},
                    "curated": True,
                },
                noise: {
                    "canonical_doc_id": noise,
                    "doc_id": noise,
                    "title": f"Noise Paper {i}",
                    "systems": {"bm25": 9, "dense": 9},
                    "curated": False,
                },
            },
        }
    }
    qrels = {
        cid: {
            cur: {
                "grade": curated_grade,
                "reason": "seed",
                "systems": ["bm25", "dense"],
                "curated": True,
            },
            noise: {
                "grade": noise_grade,
                "reason": "noise",
                "systems": ["bm25", "dense"],
                "curated": False,
            },
        }
    }
    return pool, qrels


# --------------------------------------------------------------------------- #
# (a) Clean / agreeing judge PASSES
# --------------------------------------------------------------------------- #
def test_clean_judge_passes(tmp_path: Path) -> None:
    """Judge agrees with curated truth equally across provenance → PASS.

    Curated docs are split into dense-only and bm25-only provenance; the judge
    grades them all correctly relevant (3), so disagreement is 0 on both sides.
    """
    pool_cases: dict = {}
    qrels_cases: dict = {}
    seed_ids: list[str] = []

    # Seed cases for strong Spearman/kappa (judge agrees: curated=3, noise=0).
    for i in range(10):
        pc, qc = _seed_case(i, curated_grade=3, noise_grade=0)
        pool_cases.update(pc)
        qrels_cases.update(qc)
        seed_ids.extend(pc.keys())

    # Curated docs of each provenance, all judged correctly relevant (no bias).
    for i in range(10, 30):
        cid = f"prov-case-{i}"
        is_dense = i % 2 == 0
        sysmap = {"dense": 1} if is_dense else {"bm25": 1}
        sysl = ["dense"] if is_dense else ["bm25"]
        doc = f"prov-doc-{i}"
        pool_cases[cid] = {
            "query": f"q{i}",
            "category": "crossing",
            "pdf_only": False,
            "curated_relevant": [doc],
            "pooled": {
                doc: {
                    "canonical_doc_id": doc,
                    "doc_id": doc,
                    "title": f"Paper {i}",
                    "systems": sysmap,
                    "curated": True,
                }
            },
        }
        qrels_cases[cid] = {
            doc: {"grade": 3, "reason": "rel", "systems": sysl, "curated": True}
        }

    result = judge_audit(
        str(_pool(tmp_path, pool_cases)),
        str(_qrels(tmp_path, qrels_cases)),
        str(_seeds(tmp_path, seed_ids)),
    )

    assert result["gate"] == "PASS", f"Expected PASS, got {result}"
    assert result["bias_flag"] is False
    assert result["seed_spearman"] >= 0.70
    assert result["seed_kappa"] >= 0.70


# --------------------------------------------------------------------------- #
# The FALSE-NEGATIVE fix: dense surfacing extra relevant docs is NOT bias
# --------------------------------------------------------------------------- #
def test_dense_only_relevant_docs_are_not_bias(tmp_path: Path) -> None:
    """Dense surfaces relevant docs BM25 missed → must NOT trip the gate.

    This is the exact scenario the old rate-difference metric wrongly flagged:
    many non-curated dense-only docs graded relevant, no bm25-only relevant docs.
    Because none carry curated ground truth, the corrected gate ignores them and
    PASSES — the de-biased pool's payoff is not mistaken for judge bias.
    """
    pool_cases: dict = {}
    qrels_cases: dict = {}
    seed_ids: list[str] = []

    for i in range(10):
        pc, qc = _seed_case(i, curated_grade=3, noise_grade=0)
        pool_cases.update(pc)
        qrels_cases.update(qc)
        seed_ids.extend(pc.keys())

    # Non-curated dense-only docs the judge (correctly) grades relevant, and
    # non-curated bm25-only docs it grades not-relevant. No ground truth exists
    # for these, so they must NOT be read as bias.
    for i in range(10, 40):
        cid = f"nonseed-case-{i}"
        dense_doc, bm25_doc = f"dense-doc-{i}", f"bm25-doc-{i}"
        pool_cases[cid] = {
            "query": f"q{i}",
            "category": "crossing",
            "pdf_only": False,
            "curated_relevant": [],
            "pooled": {
                dense_doc: {
                    "canonical_doc_id": dense_doc,
                    "doc_id": dense_doc,
                    "title": f"Dense {i}",
                    "systems": {"dense": 1, "colbert": 1},
                    "curated": False,
                },
                bm25_doc: {
                    "canonical_doc_id": bm25_doc,
                    "doc_id": bm25_doc,
                    "title": f"BM25 {i}",
                    "systems": {"bm25": 1},
                    "curated": False,
                },
            },
        }
        qrels_cases[cid] = {
            dense_doc: {"grade": 3, "reason": "rel", "systems": ["dense", "colbert"], "curated": False},
            bm25_doc: {"grade": 0, "reason": "no", "systems": ["bm25"], "curated": False},
        }

    result = judge_audit(
        str(_pool(tmp_path, pool_cases)),
        str(_qrels(tmp_path, qrels_cases)),
        str(_seeds(tmp_path, seed_ids)),
    )

    # No curated provenance docs outside the seed (mixed) cases → no false bias.
    assert result["bias_flag"] is False, f"False-negative regressed: {result}"
    assert result["gate"] == "PASS", f"Expected PASS, got {result}"


# --------------------------------------------------------------------------- #
# (b) CRITICAL: a genuinely biased judge STILL FAILS the gate
# --------------------------------------------------------------------------- #
def test_genuinely_biased_judge_still_caught(tmp_path: Path) -> None:
    """Judge disagrees with curated truth for ONE provenance → WARN (caught).

    Construct curated (truly-relevant) docs of both provenances. The judge grades
    the bm25-surfaced curated docs correctly relevant (3) but systematically
    grades the dense-surfaced curated docs not-relevant (0) — i.e. it penalises
    dense provenance against ground truth. Disagreement gap = 1.0 ≫ 0.25, so the
    bias must be flagged even though seed agreement is otherwise fine.
    """
    pool_cases: dict = {}
    qrels_cases: dict = {}
    seed_ids: list[str] = []

    # Seed cases agree well so seed Spearman/kappa pass; the FAIL/WARN must come
    # from the provenance arm, not from seed disagreement.
    for i in range(10):
        pc, qc = _seed_case(i, curated_grade=3, noise_grade=0)
        pool_cases.update(pc)
        qrels_cases.update(qc)
        seed_ids.extend(pc.keys())

    # Curated dense-only docs the judge WRONGLY grades 0 (truth=relevant).
    for i in range(10, 30):
        cid = f"dense-truth-case-{i}"
        doc = f"dense-truth-doc-{i}"
        pool_cases[cid] = {
            "query": f"q{i}",
            "category": "crossing",
            "pdf_only": False,
            "curated_relevant": [doc],
            "pooled": {
                doc: {
                    "canonical_doc_id": doc,
                    "doc_id": doc,
                    "title": f"Dense truth {i}",
                    "systems": {"dense": 1},
                    "curated": True,
                }
            },
        }
        qrels_cases[cid] = {
            doc: {"grade": 0, "reason": "judge wrongly rejects dense", "systems": ["dense"], "curated": True}
        }

    # Curated bm25-only docs the judge correctly grades 3 (truth=relevant).
    for i in range(30, 50):
        cid = f"bm25-truth-case-{i}"
        doc = f"bm25-truth-doc-{i}"
        pool_cases[cid] = {
            "query": f"q{i}",
            "category": "crossing",
            "pdf_only": False,
            "curated_relevant": [doc],
            "pooled": {
                doc: {
                    "canonical_doc_id": doc,
                    "doc_id": doc,
                    "title": f"BM25 truth {i}",
                    "systems": {"bm25": 1},
                    "curated": True,
                }
            },
        }
        qrels_cases[cid] = {
            doc: {"grade": 3, "reason": "judge accepts bm25", "systems": ["bm25"], "curated": True}
        }

    result = judge_audit(
        str(_pool(tmp_path, pool_cases)),
        str(_qrels(tmp_path, qrels_cases)),
        str(_seeds(tmp_path, seed_ids)),
    )

    # The biased fixture failing the gate is the load-bearing assertion.
    assert result["bias_flag"] is True, f"Biased judge slipped through: {result}"
    assert result["gate"] == "WARN", f"Expected WARN (seeds still agree), got {result}"
    assert result["dense_disagree_rate"] >= result["bm25_disagree_rate"] + 0.25
    assert result["dense_labeled_n"] > 0 and result["bm25_labeled_n"] > 0


def test_biased_judge_with_bad_seeds_fails_hard(tmp_path: Path) -> None:
    """If the judge also disagrees on seeds, the gate hard-FAILs (not just WARN)."""
    pool_cases: dict = {}
    qrels_cases: dict = {}
    seed_ids: list[str] = []

    # Inverted seed grades → poor Spearman/kappa → FAIL regardless of provenance.
    for i in range(15):
        pc, qc = _seed_case(i, curated_grade=0, noise_grade=3)
        pool_cases.update(pc)
        qrels_cases.update(qc)
        seed_ids.extend(pc.keys())

    result = judge_audit(
        str(_pool(tmp_path, pool_cases)),
        str(_qrels(tmp_path, qrels_cases)),
        str(_seeds(tmp_path, seed_ids)),
    )

    assert result["gate"] == "FAIL", f"Expected FAIL, got {result}"
    assert result["seed_spearman"] < 0.70 or result["seed_kappa"] < 0.70
