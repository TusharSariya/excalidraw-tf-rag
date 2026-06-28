#!/usr/bin/env python3
"""GATE 0 Task B — teacher (mxbai-rerank-large-v2) sanity on the catalog pool.

Reranks each catalog case's ALREADY-POOLED candidates with the cross-encoder
teacher and compares teacher nDCG@10 vs the same-pool weighted-hybrid baseline,
scored with the harness's own case_metrics (binary, threshold from qrels).

Run ON the desktop GPU:
  uv run --with mxbai-rerank python scripts/teacher_sanity_rerank.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from graph_layout_rag.eval.metrics import case_metrics
from graph_layout_rag.eval.qrels import (
    DEFAULT_RELEVANCE_THRESHOLD,
    graded_labels,
    relevant_from_grades,
)

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "data" / "eval" / "pool" / "catalog" / "pool.json"
QRELS = ROOT / "data" / "eval" / "qrels" / "catalog" / "qrels.json"
OUT_DIR = ROOT / "data" / "eval" / "teacher_sanity"
TEACHER = "mixedbread-ai/mxbai-rerank-large-v2"
HYBRID_SYSTEM = "hybrid"  # weighted-hybrid baseline ranks live under this key


def doc_text(meta: dict) -> str:
    parts = []
    if meta.get("title"):
        parts.append(str(meta["title"]))
    if meta.get("abstract"):
        parts.append(str(meta["abstract"]))
    if meta.get("excerpt"):
        parts.append(str(meta["excerpt"]))
    return "\n".join(parts).strip() or str(meta.get("doc_id") or "")


def rows_from_order(order: list[str], pooled: dict) -> list[dict]:
    """Build harness result rows (doc_id/canonical_doc_id) from a doc-id order."""
    rows = []
    for did in order:
        meta = pooled.get(did, {})
        rows.append(
            {
                "doc_id": did,
                "canonical_doc_id": meta.get("canonical_doc_id", did),
                "alias_doc_ids": meta.get("alias_doc_ids") or [],
            }
        )
    return rows


def hybrid_order(pooled: dict) -> list[str]:
    """Order pool docs by the hybrid system's pooled rank; missing -> last (stable)."""
    BIG = 10**9
    items = list(pooled.items())
    return [
        did
        for did, _ in sorted(
            items,
            key=lambda kv: kv[1].get("systems", {}).get(HYBRID_SYSTEM, BIG),
        )
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pool = json.loads(POOL.read_text())
    qrels = json.loads(QRELS.read_text())
    threshold = int(qrels.get("relevance_threshold", DEFAULT_RELEVANCE_THRESHOLD))
    labels = graded_labels(qrels)  # case_id -> {doc_id: grade}

    cases = pool["cases"]
    print(
        f"pool: track={pool['track']} embed_profile={pool['embed_profile']} "
        f"depth={pool['depth']} cases={len(cases)} threshold>={threshold}",
        flush=True,
    )

    print(f"loading teacher {TEACHER} on cuda ...", flush=True)
    from mxbai_rerank import MxbaiRerankV2

    t0 = time.time()
    # 8 GB card: cap context + small batch to avoid OOM on long excerpts.
    model = MxbaiRerankV2(
        TEACHER, device="cuda", torch_dtype=torch.float16, max_length=512
    )
    print(f"teacher loaded in {time.time()-t0:.1f}s", flush=True)

    per_case = []
    teacher_ndcgs = []
    hybrid_ndcgs = []
    skipped = []

    for i, (case_id, case) in enumerate(sorted(cases.items())):
        query = case["query"]
        pooled = case["pooled"]
        grades = labels.get(case_id, {})
        relevant = relevant_from_grades(grades, threshold=threshold)
        # also union curated_relevant (harness overlays them as relevant)
        relevant = set(relevant) | set(case.get("curated_relevant") or [])
        if not relevant:
            skipped.append(case_id)
            continue

        doc_ids = list(pooled.keys())
        docs = [doc_text(pooled[d]) for d in doc_ids]

        ranked = model.rank(
            query, docs, top_k=len(docs), batch_size=8, return_documents=False
        )
        teacher_order = [doc_ids[r.index] for r in ranked]

        t_rows = rows_from_order(teacher_order, pooled)
        h_rows = rows_from_order(hybrid_order(pooled), pooled)
        t_m = case_metrics(t_rows, relevant)
        h_m = case_metrics(h_rows, relevant)

        teacher_ndcgs.append(t_m["ndcg@10"])
        hybrid_ndcgs.append(h_m["ndcg@10"])
        per_case.append(
            {
                "case_id": case_id,
                "category": case.get("category"),
                "n_pool": len(doc_ids),
                "n_relevant": len(relevant),
                "teacher_ndcg@10": t_m["ndcg@10"],
                "hybrid_ndcg@10": h_m["ndcg@10"],
                "teacher_recall@10": t_m["recall@10"],
                "hybrid_recall@10": h_m["recall@10"],
            }
        )
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(
            f"[{i+1}/{len(cases)}] {case_id:48s} "
            f"teacher={t_m['ndcg@10']:.4f} hybrid={h_m['ndcg@10']:.4f} "
            f"pool={len(doc_ids)} rel={len(relevant)} gpu_peak={mem:.2f}GB",
            flush=True,
        )

    n = len(teacher_ndcgs)
    summary = {
        "teacher_model": TEACHER,
        "pool": str(POOL.relative_to(ROOT)),
        "qrels": str(QRELS.relative_to(ROOT)),
        "embed_profile_of_pool": pool["embed_profile"],
        "relevance_threshold": threshold,
        "n_cases_scored": n,
        "n_cases_skipped_no_relevant": len(skipped),
        "skipped_cases": skipped,
        "teacher_mean_ndcg@10": sum(teacher_ndcgs) / n if n else None,
        "hybrid_mean_ndcg@10": sum(hybrid_ndcgs) / n if n else None,
        "delta_teacher_minus_hybrid": (
            (sum(teacher_ndcgs) - sum(hybrid_ndcgs)) / n if n else None
        ),
        "teacher_wins": sum(1 for c in per_case if c["teacher_ndcg@10"] > c["hybrid_ndcg@10"] + 1e-9),
        "hybrid_wins": sum(1 for c in per_case if c["hybrid_ndcg@10"] > c["teacher_ndcg@10"] + 1e-9),
        "ties": sum(1 for c in per_case if abs(c["teacher_ndcg@10"] - c["hybrid_ndcg@10"]) <= 1e-9),
        "gpu_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
        "gpu_device": torch.cuda.get_device_name(0),
    }
    (OUT_DIR / "per_case.json").write_text(json.dumps(per_case, indent=2))
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
