#!/usr/bin/env python3
"""Analyze benchmark run JSON files from data/eval/runs/.

Produces:
1. Per-set strategy ranking tables (by qrels path substring filter)
2. Paired bootstrap CIs for model-vs-model deltas
3. A judge_audit gate result

Usage:
    python scripts/analyze_bakeoff.py [--runs-dir PATH] [--keyword-qrels STR]
        [--nl-qrels STR] [--pool PATH] [--seeds PATH]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_runs(runs_dir: str = "data/eval/runs") -> list[dict]:
    """Load all JSON run files from a runs directory.

    Each run directory contains a ``strategies/`` subdirectory with per-strategy
    JSON files. Each file has the following keys (subset relevant here):
        strategy          (str)
        track             (str)
        ndcg@10           (float)
        recall@10         (float)
        cases             (list[dict])  — each case has 'id' and 'ndcg@10'

    The loader synthesises a ``per_case_ndcg`` dict (case_id -> float) from
    the cases list so callers can treat every run uniformly.

    The ``qrels_path`` key is not stored inside individual strategy files; it is
    inferred from the benchmark-level metadata file when present (``benchmark.json``
    inside the run dir), or left as ``""`` when absent.
    """
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        return []

    all_runs: list[dict] = []

    for run_dir in sorted(runs_path.iterdir()):
        if not run_dir.is_dir():
            continue

        # Read optional top-level benchmark metadata (contains qrels path)
        benchmark_meta: dict[str, Any] = {}
        bench_file = run_dir / "benchmark.json"
        if bench_file.is_file():
            try:
                benchmark_meta = json.loads(bench_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

        qrels_path = benchmark_meta.get("qrels", "") or ""
        embed_profile = benchmark_meta.get("embed_profile", "") or run_dir.name

        strategies_dir = run_dir / "strategies"
        if not strategies_dir.is_dir():
            continue

        for strat_file in sorted(strategies_dir.iterdir()):
            if strat_file.suffix != ".json":
                continue
            try:
                payload = json.loads(strat_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            # Skip failed / memory-aborted runs
            if payload.get("status") in ("failed", "memory_aborted"):
                continue
            if "ndcg@10" not in payload:
                continue

            # Build per_case_ndcg
            per_case: dict[str, float] = {}
            for case in payload.get("cases", []):
                case_id = case.get("id") or case.get("case_id")
                ndcg_val = case.get("ndcg@10")
                if case_id is not None and ndcg_val is not None:
                    per_case[case_id] = float(ndcg_val)

            all_runs.append(
                {
                    "run_dir": str(run_dir),
                    "embed_profile": payload.get("embed_profile", embed_profile),
                    "qrels_path": qrels_path,
                    "strategy": payload.get("strategy", ""),
                    "track": payload.get("track", ""),
                    "ndcg10": float(payload["ndcg@10"]),
                    "recall10": float(payload.get("recall@10", 0.0)),
                    "per_case_ndcg": per_case,
                }
            )

    return all_runs


# ---------------------------------------------------------------------------
# Strategy ranking
# ---------------------------------------------------------------------------

def load_leave_dense_out_ids(path: str) -> set[str]:
    """Load the leave-dense-out gold case-id allowlist.

    Accepts either the sidecar emitted by ``eval pool --leave-dense-out``
    (``pool.leave_dense_out.json`` with a ``case_ids`` list) or a bare JSON list
    of case IDs.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return set(raw.get("case_ids", []))
    if isinstance(raw, list):
        return set(raw)
    return set()


def restrict_runs_to_cases(runs: list[dict], keep_ids: set[str]) -> list[dict]:
    """Restrict every run's per-case scores to ``keep_ids`` and recompute nDCG@10.

    Recomputes ``ndcg10`` as the mean over the surviving cases so that ranking
    reflects only the leave-dense-out gold subset (cases whose seed a lexical
    system surfaced). Runs that share no case with ``keep_ids`` are dropped, since
    they carry no signal on this subset. ``recall10`` is left unchanged (it is not
    re-derivable from per-case nDCG); callers should rely on ``mean_ndcg10`` here.
    """
    restricted: list[dict] = []
    for run in runs:
        kept = {
            cid: v for cid, v in run.get("per_case_ndcg", {}).items() if cid in keep_ids
        }
        if not kept:
            continue
        new_run = dict(run)
        new_run["per_case_ndcg"] = kept
        new_run["ndcg10"] = float(mean(kept.values()))
        restricted.append(new_run)
    return restricted


def rank_strategies(runs: list[dict], qrels_path_filter: str) -> list[dict]:
    """Filter runs by qrels path substring, group by strategy, rank by mean nDCG@10.

    Args:
        runs: List of run dicts as returned by :func:`load_runs`.
        qrels_path_filter: Substring that must appear in ``qrels_path`` to
            include a run.  Pass ``""`` to include all runs.

    Returns:
        List of ``{strategy, mean_ndcg10, mean_recall10, n_cases}`` sorted
        descending by *mean_ndcg10*.
    """
    filtered = [
        r for r in runs
        if qrels_path_filter in r.get("qrels_path", "")
    ]

    by_strategy: dict[str, list[dict]] = {}
    for run in filtered:
        s = run["strategy"]
        by_strategy.setdefault(s, []).append(run)

    ranked: list[dict] = []
    for strategy, group in by_strategy.items():
        all_ndcg = [r["ndcg10"] for r in group]
        all_recall = [r["recall10"] for r in group]
        # Aggregate per-case counts across all runs in the group
        n_cases = sum(len(r["per_case_ndcg"]) for r in group)
        ranked.append(
            {
                "strategy": strategy,
                "mean_ndcg10": round(mean(all_ndcg), 6),
                "mean_recall10": round(mean(all_recall), 6),
                "n_cases": n_cases,
            }
        )

    ranked.sort(key=lambda x: x["mean_ndcg10"], reverse=True)
    return ranked


# ---------------------------------------------------------------------------
# Paired bootstrap CI
# ---------------------------------------------------------------------------

def paired_bootstrap_ci(
    run_a: dict,
    run_b: dict,
    n_resamples: int = 1000,
    alpha: float = 0.05,
) -> dict:
    """Paired bootstrap over per_case_ndcg.

    Args:
        run_a: Run dict with ``per_case_ndcg`` key.
        run_b: Run dict with ``per_case_ndcg`` key.
        n_resamples: Number of bootstrap resamples.
        alpha: Significance level for the CI (two-sided).

    Returns:
        ``{delta_mean, ci_low, ci_high, contains_zero, label}`` where
        *label* is ``"indistinguishable"`` when the CI contains zero.
    """
    cases_a = run_a.get("per_case_ndcg", {})
    cases_b = run_b.get("per_case_ndcg", {})

    # Use intersection of cases for paired comparison
    common_ids = sorted(set(cases_a) & set(cases_b))
    if not common_ids:
        return {
            "delta_mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "contains_zero": True,
            "label": "indistinguishable (no common cases)",
        }

    deltas = np.array(
        [cases_a[cid] - cases_b[cid] for cid in common_ids], dtype=float
    )
    delta_mean = float(np.mean(deltas))

    rng = np.random.default_rng(42)
    n = len(deltas)
    boot_means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        boot_means[i] = np.mean(deltas[indices])

    lo_pct = (alpha / 2) * 100
    hi_pct = (1 - alpha / 2) * 100
    ci_low = float(np.percentile(boot_means, lo_pct))
    ci_high = float(np.percentile(boot_means, hi_pct))
    contains_zero = ci_low <= 0.0 <= ci_high

    return {
        "delta_mean": round(delta_mean, 6),
        "ci_low": round(ci_low, 6),
        "ci_high": round(ci_high, 6),
        "contains_zero": contains_zero,
        "label": "indistinguishable" if contains_zero else (
            f"{'A wins' if delta_mean > 0 else 'B wins'} "
            f"(Δ={delta_mean:+.4f}, CI=[{ci_low:.4f}, {ci_high:.4f}])"
        ),
    }


# ---------------------------------------------------------------------------
# Judge audit
# ---------------------------------------------------------------------------

def judge_audit(
    pool_path: str,
    qrels_path: str,
    seed_cases_path: str | None = None,
) -> dict:
    """Judge-bias gate.

    Checks two things:
    1. **Seed Spearman/κ**: If *seed_cases_path* is given, compares judge scores
       on curated seed cases against the curated relevance labels.  Seed cases are
       curated as relevant, so the judge should agree (high Spearman r and Cohen κ).
    2. **Provenance-conditioned disagreement** (the genuine judge-bias signal):
       The judge is *source-blind* — its prompt never reveals which retrieval
       system surfaced a doc (see ``eval/judge.py::build_judge_prompt``). So a
       higher raw relevance rate among dense-only docs is NOT evidence of judge
       bias — it is the designed payoff of de-biased pooling (dense surfaces
       genuinely-relevant docs BM25 missed). Judge bias can only manifest as the
       judge *disagreeing with ground truth differently depending on provenance*.
       We therefore measure, over docs that carry a curated ground-truth label,
       the judge↔truth disagreement rate split by provenance (dense-surfaced vs
       bm25-surfaced) and flag bias when the gap exceeds 0.25.

    Gate logic:
        PASS  — seed metrics >= 0.70 AND bias_flag is False
        WARN  — seed metrics OK but bias_flag is True
        FAIL  — seed metrics < 0.70

    Args:
        pool_path: Path to ``pool.json``.
        qrels_path: Path to ``qrels.json``.
        seed_cases_path: Optional JSON file listing case IDs treated as seeds
            (list of str).

    Returns:
        Dict with keys: seed_spearman, seed_kappa, dense_disagree_rate,
        bm25_disagree_rate, dense_labeled_n, bm25_labeled_n, bias_flag, gate.
    """
    # --- Load pool -------------------------------------------------------
    pool_data: dict[str, Any] = json.loads(Path(pool_path).read_text(encoding="utf-8"))
    pool_cases: dict[str, Any] = pool_data.get("cases", {})

    # --- Load qrels ------------------------------------------------------
    qrels_data: dict[str, Any] = json.loads(Path(qrels_path).read_text(encoding="utf-8"))
    # qrels_data["cases"] is dict: case_id -> {doc_id -> {grade, systems, curated, ...}}
    qrels_cases: dict[str, Any] = qrels_data.get("cases", {})

    # --- Resolve seed case IDs -------------------------------------------
    seed_case_ids: set[str] = set()
    if seed_cases_path:
        raw = json.loads(Path(seed_cases_path).read_text(encoding="utf-8"))
        seed_case_ids = set(raw) if isinstance(raw, list) else set(raw.keys())

    # If no seed file, treat curated=True docs across all qrels cases as seeds
    if not seed_case_ids:
        for case_id, docs in qrels_cases.items():
            if isinstance(docs, dict):
                if any(d.get("curated", False) for d in docs.values()):
                    seed_case_ids.add(case_id)

    # ------------------------------------------------------------------
    # 1. Seed Spearman / Cohen κ
    # ------------------------------------------------------------------
    seed_judge_scores: list[float] = []
    seed_curated_grades: list[float] = []

    for case_id in seed_case_ids:
        docs = qrels_cases.get(case_id, {})
        if not isinstance(docs, dict):
            continue
        for doc_id, info in docs.items():
            if not isinstance(info, dict):
                continue
            grade = info.get("grade")
            if grade is None:
                continue
            # Curated docs are relevant by construction → curated grade = 3
            curated_grade = 3.0 if info.get("curated", False) else 0.0
            seed_judge_scores.append(float(grade))
            seed_curated_grades.append(curated_grade)

    if len(seed_judge_scores) >= 2:
        spearman_r, _ = stats.spearmanr(seed_curated_grades, seed_judge_scores)
        spearman_r = float(spearman_r) if not np.isnan(spearman_r) else 0.0
        # Cohen κ on binarised labels (grade >= 2 = relevant)
        judge_bin = [1 if s >= 2 else 0 for s in seed_judge_scores]
        curated_bin = [1 if g >= 2 else 0 for g in seed_curated_grades]
        kappa = _cohens_kappa(curated_bin, judge_bin)
    else:
        spearman_r = 0.0
        kappa = 0.0

    # ------------------------------------------------------------------
    # 2. Provenance-conditioned judge↔truth DISAGREEMENT
    # ------------------------------------------------------------------
    # FIX (was a false-negative): the previous version compared the raw
    # *relevance rate* of dense-only vs bm25-only docs and flagged bias when
    # dense scored >0.25 higher. But a dense-only doc the judge graded relevant
    # is NOT judge bias — the judge is source-blind, so it cannot favour dense by
    # provenance; a relevant dense-only doc is exactly the de-biased pool's payoff
    # (a genuinely-relevant doc BM25 missed). Counting those as "bias" inflated
    # the metric and could wrongly WARN/FAIL on a perfectly fair judge.
    #
    # The only provenance signal that *is* judge bias is the judge disagreeing
    # with GROUND TRUTH in a provenance-dependent way. We have trustworthy truth
    # only for curated docs (curated=True ⇒ relevant by construction, exactly the
    # truth the seed-Spearman arm uses). So we measure, over curated-labelled docs,
    # the judge↔truth disagreement rate split by which retrieval family surfaced
    # the doc, and flag bias when that gap exceeds 0.25. A fair judge disagrees
    # with truth at the same rate regardless of provenance (gap ~0); a judge that
    # systematically penalises one provenance's docs relative to their true
    # relevance shows up as a large gap and is still caught.
    def _provenance(systems: Any) -> str | None:
        # ``systems`` is a dict (pool: system->rank) or a list (qrels). Membership
        # works for both. Returns "dense", "bm25", or None (mixed/other → skip:
        # a mixed-provenance doc carries no clean provenance signal).
        has_dense = any(s in systems for s in ("dense", "colbert"))
        has_lexical = any(s in systems for s in ("bm25", "splade"))
        if has_dense and not has_lexical:
            return "dense"
        if has_lexical and not has_dense:
            return "bm25"
        return None

    # Collect curated-labelled docs across ALL cases (curated truth = relevant).
    # Provenance is read from the pool's system-provenance map; fall back to the
    # qrels ``systems`` list when the doc is not present in the pool.
    dense_disagree = dense_labeled = 0
    bm25_disagree = bm25_labeled = 0

    for case_id, docs in qrels_cases.items():
        if not isinstance(docs, dict):
            continue
        pooled_docs = pool_cases.get(case_id, {}).get("pooled", {})
        for doc_id, info in docs.items():
            if not isinstance(info, dict) or not info.get("curated", False):
                continue
            grade = info.get("grade")
            if grade is None:
                continue
            pool_entry = pooled_docs.get(doc_id, {})
            systems = pool_entry.get("systems") if isinstance(pool_entry, dict) else None
            if not systems:
                systems = info.get("systems", [])
            prov = _provenance(systems)
            if prov is None:
                continue
            # Curated ⇒ truly relevant; the judge disagrees if it grades < 2.
            disagrees = 1 if grade < 2 else 0
            if prov == "dense":
                dense_labeled += 1
                dense_disagree += disagrees
            else:
                bm25_labeled += 1
                bm25_disagree += disagrees

    dense_rate = dense_disagree / dense_labeled if dense_labeled > 0 else 0.0
    bm25_rate = bm25_disagree / bm25_labeled if bm25_labeled > 0 else 0.0
    # Two-sided: bias in EITHER direction (judge penalises one provenance more).
    bias_flag = abs(dense_rate - bm25_rate) > 0.25

    # ------------------------------------------------------------------
    # Gate decision
    # ------------------------------------------------------------------
    seed_ok = spearman_r >= 0.70 and kappa >= 0.70
    if not seed_ok:
        gate = "FAIL"
    elif bias_flag:
        gate = "WARN"
    else:
        gate = "PASS"

    return {
        "seed_spearman": round(spearman_r, 4),
        "seed_kappa": round(kappa, 4),
        "dense_disagree_rate": round(dense_rate, 4),
        "bm25_disagree_rate": round(bm25_rate, 4),
        "dense_labeled_n": dense_labeled,
        "bm25_labeled_n": bm25_labeled,
        "bias_flag": bias_flag,
        "gate": gate,
    }


def _cohens_kappa(y_true: list[int], y_pred: list[int]) -> float:
    """Simple binary Cohen κ."""
    n = len(y_true)
    if n == 0:
        return 0.0
    observed_agreement = sum(a == b for a, b in zip(y_true, y_pred)) / n
    p_true_pos = sum(y_true) / n
    p_pred_pos = sum(y_pred) / n
    p_chance = (p_true_pos * p_pred_pos) + ((1 - p_true_pos) * (1 - p_pred_pos))
    if p_chance >= 1.0:
        return 1.0
    return (observed_agreement - p_chance) / (1 - p_chance)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _print_ranking_table(label: str, ranked: list[dict]) -> None:
    if not ranked:
        print(f"  (no runs match filter '{label}')")
        return
    col_w = max(len(r["strategy"]) for r in ranked) + 2
    header = f"{'strategy':<{col_w}}  {'nDCG@10':>9}  {'Recall@10':>10}  {'n_cases':>8}"
    print(header)
    print("-" * len(header))
    for row in ranked:
        print(
            f"{row['strategy']:<{col_w}}  {row['mean_ndcg10']:>9.4f}  "
            f"{row['mean_recall10']:>10.4f}  {row['n_cases']:>8}"
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze graph-layout-rag benchmark runs."
    )
    parser.add_argument(
        "--runs-dir",
        default="data/eval/runs",
        help="Directory containing run subdirectories (default: data/eval/runs).",
    )
    parser.add_argument(
        "--keyword-qrels",
        default="",
        help="Substring filter for the keyword/catalog qrels path (default: '' = all).",
    )
    parser.add_argument(
        "--nl-qrels",
        default="pdf-deep-read",
        help="Substring filter for the NL/pdf-deep-read qrels path.",
    )
    parser.add_argument("--pool", default=None, help="Path to pool.json for judge_audit.")
    parser.add_argument("--seeds", default=None, help="Path to seed cases JSON for judge_audit.")
    parser.add_argument(
        "--leave-dense-out",
        default=None,
        help=(
            "Path to pool.leave_dense_out.json (or a bare case-id list). When given, "
            "all rankings/CIs are recomputed on ONLY the gold cases whose seed a "
            "lexical/BM25-family system surfaced — removing the dense/HyDE "
            "survivorship circularity from the BM25-falls comparison."
        ),
    )
    args = parser.parse_args()

    runs = load_runs(args.runs_dir)
    if not runs:
        print(f"No run files found in '{args.runs_dir}'. Run a benchmark first.")
        return

    print(f"\nLoaded {len(runs)} strategy run(s) from '{args.runs_dir}'.\n")

    if args.leave_dense_out:
        keep_ids = load_leave_dense_out_ids(args.leave_dense_out)
        runs = restrict_runs_to_cases(runs, keep_ids)
        print(
            f"[leave-dense-out] restricted to {len(keep_ids)} lexical-seeded gold "
            f"case(s); {len(runs)} run(s) retain at least one such case.\n"
            "  Rankings/CIs below are computed on this subset only.\n"
        )
        if not runs:
            print("[leave-dense-out] no runs overlap the subset; nothing to rank.")
            return

    # --- Keyword / catalog ranking ---
    print("=== Strategy Ranking (keyword qrels filter: '{}') ===".format(args.keyword_qrels))
    kw_ranked = rank_strategies(runs, args.keyword_qrels)
    _print_ranking_table(args.keyword_qrels, kw_ranked)

    # --- NL / pdf-deep-read ranking ---
    print("\n=== Strategy Ranking (NL qrels filter: '{}') ===".format(args.nl_qrels))
    nl_ranked = rank_strategies(runs, args.nl_qrels)
    _print_ranking_table(args.nl_qrels, nl_ranked)

    # --- Pairwise CIs for top-2 strategies (if available) ---
    if len(kw_ranked) >= 2:
        print("\n=== Paired Bootstrap CI: top-2 strategies (keyword set) ===")
        top_a_name = kw_ranked[0]["strategy"]
        top_b_name = kw_ranked[1]["strategy"]
        # Merge per-case scores across runs for each strategy
        a_runs = [r for r in runs if r["strategy"] == top_a_name]
        b_runs = [r for r in runs if r["strategy"] == top_b_name]
        if a_runs and b_runs:
            merged_a = {k: v for r in a_runs for k, v in r["per_case_ndcg"].items()}
            merged_b = {k: v for r in b_runs for k, v in r["per_case_ndcg"].items()}
            ci = paired_bootstrap_ci(
                {"per_case_ndcg": merged_a},
                {"per_case_ndcg": merged_b},
            )
            print(f"  A: {top_a_name}  vs  B: {top_b_name}")
            print(f"  {ci['label']}")
            print(
                f"  Δmean={ci['delta_mean']:+.4f}  "
                f"95% CI=[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]"
            )

    # --- Judge audit (optional) ---
    if args.pool:
        qrels_for_audit = args.keyword_qrels or args.nl_qrels or ""
        # Attempt to resolve a real qrels file path
        qrels_candidate = Path(qrels_for_audit) if qrels_for_audit else None
        if qrels_candidate and not qrels_candidate.is_file():
            # Try to find catalog qrels
            qrels_candidate = Path("data/eval/qrels/catalog/qrels.json")
        if qrels_candidate and qrels_candidate.is_file():
            print("\n=== Judge Audit ===")
            result = judge_audit(
                pool_path=args.pool,
                qrels_path=str(qrels_candidate),
                seed_cases_path=args.seeds,
            )
            print(f"  Gate:                  {result['gate']}")
            print(f"  Seed Spearman r:       {result['seed_spearman']:.4f}")
            print(f"  Seed Cohen κ:          {result['seed_kappa']:.4f}")
            print(
                f"  Dense disagree-w-truth: {result['dense_disagree_rate']:.4f}"
                f"  (n={result['dense_labeled_n']})"
            )
            print(
                f"  BM25 disagree-w-truth:  {result['bm25_disagree_rate']:.4f}"
                f"  (n={result['bm25_labeled_n']})"
            )
            print(f"  Bias flag:             {result['bias_flag']}")
            if result["gate"] == "WARN":
                print(
                    "  WARNING: Seed metrics pass but the judge disagrees with "
                    "curated ground truth >0.25 more for one retrieval provenance "
                    "than the other — judge may be provenance-biased."
                )
            elif result["gate"] == "FAIL":
                print(
                    "  FAIL: Seed agreement below 0.70 — judge scores are not "
                    "reliable. Do not use these qrels."
                )
        else:
            print("\n[judge_audit] Skipped: --pool given but no resolvable qrels file.")
    else:
        print("\n[judge_audit] Skipped: pass --pool to enable.")


if __name__ == "__main__":
    main()
