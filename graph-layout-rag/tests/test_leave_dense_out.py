"""Tests for the leave-dense-out pool + per-case seed-system attribution.

Survivorship circularity these guard against
--------------------------------------------
A synthetic NL gold case's only positive is its *seed* doc; the case is only a
useful gold case if the pooled judge graded that seed relevant. But the standard
pool includes dense + HyDE systems, so a seed can be surfaced (and thus end up in
the judged set) BECAUSE dense found it. The bake-off then "discovers" dense beats
BM25 on exactly those cases — partly tautological.

``seed_attribution`` tags every case with which pool system surfaced its seed and
a lexical-vs-dense provenance; ``leave_dense_out_case_ids`` keeps only the cases a
lexical/BM25-family system surfaced, so the BM25-falls comparison can be recomputed
on a subset whose existence does NOT depend on dense retrieval.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from graph_layout_rag.eval.pooling import (
    DENSE_NEURAL_SEED_SYSTEMS,
    LEXICAL_SEED_SYSTEMS,
    attribution_breakdown,
    classify_system,
    leave_dense_out_case_ids,
    seed_attribution,
)

# scripts/ importable for analyze_bakeoff restrict helpers.
_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from analyze_bakeoff import (  # noqa: E402
    load_leave_dense_out_ids,
    rank_strategies,
    restrict_runs_to_cases,
)


# --------------------------------------------------------------------------- #
# System classification
# --------------------------------------------------------------------------- #
def test_classification_partitions_systems() -> None:
    assert classify_system("bm25") == "lexical"
    for s in ("dense", "colbert", "hybrid", "hyde", "multi_query", "splade"):
        assert classify_system(s) == "dense", s
    # bm25 is the only lexical-by-default system; dense/neural set excludes it.
    assert "bm25" in LEXICAL_SEED_SYSTEMS
    assert "bm25" not in DENSE_NEURAL_SEED_SYSTEMS
    # Unknown systems are conservatively treated as dense/neural (not lexical).
    assert classify_system("some_new_neural_thing") == "dense"


# --------------------------------------------------------------------------- #
# Pool fixture: 4 synthetic cases with distinct seed provenance
# --------------------------------------------------------------------------- #
def _pool_with_seeds() -> tuple[dict, dict]:
    """Return (pool_payload, case_seed) with one case per provenance class."""
    cases = {
        # seed surfaced by bm25 (+dense) → lexical provenance
        "synth-lex": {
            "pooled": {
                "seed-lex": {"systems": {"bm25": 1, "dense": 3}},
                "other": {"systems": {"dense": 1}},
            }
        },
        # seed surfaced ONLY by dense/hyde → dense provenance (circularity-prone)
        "synth-dense": {
            "pooled": {
                "seed-dense": {"systems": {"dense": 1, "hyde": 2}},
            }
        },
        # seed surfaced ONLY by hyde → dense provenance
        "synth-hyde": {
            "pooled": {
                "seed-hyde": {"systems": {"hyde": 4}},
            }
        },
        # seed folded in as curated only, no retrieval system surfaced it → none
        "synth-none": {
            "pooled": {
                "seed-none": {"systems": {}},
            }
        },
    }
    case_seed = {
        "synth-lex": "seed-lex",
        "synth-dense": "seed-dense",
        "synth-hyde": "seed-hyde",
        "synth-none": "seed-none",
    }
    return {"cases": cases}, case_seed


def test_attribution_tags_provenance() -> None:
    pool, case_seed = _pool_with_seeds()
    attr = seed_attribution(pool, case_seed)

    assert attr["synth-lex"]["provenance"] == "lexical"
    assert attr["synth-lex"]["lexical"] == ["bm25"]
    assert "dense" in attr["synth-lex"]["dense"]

    assert attr["synth-dense"]["provenance"] == "dense"
    assert attr["synth-dense"]["lexical"] == []
    assert set(attr["synth-dense"]["dense"]) == {"dense", "hyde"}

    assert attr["synth-hyde"]["provenance"] == "dense"
    assert attr["synth-hyde"]["dense"] == ["hyde"]

    assert attr["synth-none"]["provenance"] == "none"
    assert attr["synth-none"]["systems"] == []


def test_attribution_handles_list_systems() -> None:
    """qrels-style ``systems`` lists are accepted, not just pool-style dicts."""
    pool = {"cases": {"c": {"pooled": {"s": {"systems": ["bm25", "dense"]}}}}}
    attr = seed_attribution(pool, {"c": "s"})
    assert attr["c"]["provenance"] == "lexical"
    assert attr["c"]["lexical"] == ["bm25"]


def test_breakdown_counts() -> None:
    pool, case_seed = _pool_with_seeds()
    bd = attribution_breakdown(seed_attribution(pool, case_seed))
    assert bd["n_cases"] == 4
    assert bd["lexical_seeded"] == 1
    assert bd["dense_only_seeded"] == 2
    assert bd["unseeded"] == 1


# --------------------------------------------------------------------------- #
# Leave-dense-out keeps ONLY lexical-seeded cases (excludes dense/HyDE)
# --------------------------------------------------------------------------- #
def test_leave_dense_out_excludes_dense_and_hyde() -> None:
    pool, case_seed = _pool_with_seeds()
    keep = leave_dense_out_case_ids(seed_attribution(pool, case_seed))
    assert keep == ["synth-lex"]
    # the dense/hyde/none-seeded cases are all dropped
    assert "synth-dense" not in keep
    assert "synth-hyde" not in keep
    assert "synth-none" not in keep


def test_leave_dense_out_requires_corroboration_when_graded() -> None:
    """With qrels, a lexical-seeded case still drops if the judge graded seed < 2."""
    pool, case_seed = _pool_with_seeds()
    # Add a second lexical-seeded case whose seed the judge graded 1 (not relevant).
    pool["cases"]["synth-lex2"] = {"pooled": {"seed-lex2": {"systems": {"bm25": 1}}}}
    case_seed["synth-lex2"] = "seed-lex2"
    qrels = {
        "cases": {
            "synth-lex": {"seed-lex": {"grade": 3, "systems": ["bm25"]}},
            "synth-lex2": {"seed-lex2": {"grade": 1, "systems": ["bm25"]}},
        }
    }
    attr = seed_attribution(pool, case_seed, qrels=qrels)
    assert attr["synth-lex"]["seed_corroborated"] is True
    assert attr["synth-lex2"]["seed_corroborated"] is False

    keep = leave_dense_out_case_ids(attr, require_corroborated=True)
    assert keep == ["synth-lex"]  # lex2 dropped: lexical-seeded but judged < 2
    # Disabling the corroboration requirement keeps both lexical-seeded cases.
    keep_all = leave_dense_out_case_ids(attr, require_corroborated=False)
    assert set(keep_all) == {"synth-lex", "synth-lex2"}


# --------------------------------------------------------------------------- #
# analyze_bakeoff: recompute the BM25-falls comparison on the subset
# --------------------------------------------------------------------------- #
def test_restrict_runs_recomputes_ndcg_on_subset() -> None:
    """Restricting to lexical-seeded cases changes the bm25-vs-dense verdict.

    Construct a tautological win: dense beats bm25 only on the dense-only-seeded
    case. After leave-dense-out (keep only the lexical case), the two strategies
    tie — demonstrating the headline was driven by the circular cases.
    """
    runs = [
        {
            "strategy": "bm25",
            "qrels_path": "catalog-nl",
            "recall10": 0.0,
            "per_case_ndcg": {"synth-lex": 0.80, "synth-dense": 0.20},
        },
        {
            "strategy": "dense",
            "qrels_path": "catalog-nl",
            "recall10": 0.0,
            "per_case_ndcg": {"synth-lex": 0.80, "synth-dense": 0.95},
        },
    ]
    # Full set: dense wins (mean 0.875 > 0.50).
    full = {r["strategy"]: r for r in rank_strategies(_with_ndcg(runs), "catalog-nl")}
    assert full["dense"]["mean_ndcg10"] > full["bm25"]["mean_ndcg10"]

    # Leave-dense-out: keep only the lexical-seeded case → tie.
    restricted = restrict_runs_to_cases(runs, {"synth-lex"})
    sub = {r["strategy"]: r for r in rank_strategies(restricted, "catalog-nl")}
    assert sub["dense"]["mean_ndcg10"] == sub["bm25"]["mean_ndcg10"] == 0.80
    assert sub["dense"]["n_cases"] == 1


def test_restrict_drops_runs_with_no_overlap() -> None:
    runs = [
        {
            "strategy": "bm25",
            "qrels_path": "x",
            "recall10": 0.0,
            "per_case_ndcg": {"a": 0.5},
        }
    ]
    assert restrict_runs_to_cases(runs, {"b"}) == []


def test_load_leave_dense_out_ids_sidecar_and_list(tmp_path: Path) -> None:
    sidecar = tmp_path / "pool.leave_dense_out.json"
    sidecar.write_text(json.dumps({"case_ids": ["x", "y"]}), encoding="utf-8")
    assert load_leave_dense_out_ids(str(sidecar)) == {"x", "y"}

    bare = tmp_path / "ids.json"
    bare.write_text(json.dumps(["p", "q"]), encoding="utf-8")
    assert load_leave_dense_out_ids(str(bare)) == {"p", "q"}


def _with_ndcg(runs: list[dict]) -> list[dict]:
    """Mirror load_runs: derive ndcg10 as the mean of per_case_ndcg for ranking."""
    from statistics import mean

    out = []
    for r in runs:
        rr = dict(r)
        rr["ndcg10"] = float(mean(r["per_case_ndcg"].values()))
        out.append(rr)
    return out
