#!/usr/bin/env python3
"""GATE 0 follow-up — fuller-text re-test for the best reranker.

Instead of scoring query x short pool-excerpt, score query x the candidate's
FULL chunk text (all chunks of the doc, max-pooled) pulled from the
gemini-2-structure-v1 LanceDB (the pool's own index). Same 49 catalog cases,
same same-pool weighted-hybrid baseline, same binary case_metrics nDCG@10.

Rules out "we under-sold the teacher by 150-token excerpt truncation".

Usage (desktop):
  HF_HUB_DISABLE_XET=1 uv run python scripts/teacher_fulltext_retest.py \
      --backend qwen3 --model Qwen/Qwen3-Reranker-0.6B --tag qwen3-0.6b-fulltext
"""
from __future__ import annotations

import argparse
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
from graph_layout_rag.paths import CHUNKS_TABLE, profile_index_paths

# reuse the backends from the bakeoff module
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "teacher_bakeoff", Path(__file__).resolve().parent / "teacher_bakeoff.py"
)
_bk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bk)

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "data" / "eval" / "pool" / "catalog" / "pool.json"
QRELS = ROOT / "data" / "eval" / "qrels" / "catalog" / "qrels.json"
OUT_DIR = ROOT / "data" / "eval" / "teacher_sanity" / "bakeoff"
POOL_PROFILE = "gemini-2-structure-v1"
MAX_CHARS = 4000  # cap per-chunk text fed to the reranker (truncated again by tokenizer)


def load_chunks_by_doc(profile: str) -> dict[str, list[str]]:
    import lancedb

    paths = profile_index_paths(profile)
    db = lancedb.connect(str(paths.lance_dir))
    rows = (
        db.open_table(CHUNKS_TABLE)
        .to_arrow()
        .select(["doc_id", "text", "title"])
        .to_pylist()
    )
    by_doc: dict[str, list[str]] = {}
    for r in rows:
        did = r.get("doc_id")
        if not did:
            continue
        title = r.get("title") or ""
        text = r.get("text") or ""
        combined = (f"{title}\n{text}".strip())[:MAX_CHARS]
        by_doc.setdefault(did, []).append(combined)
    return by_doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["ce", "mxbai", "qwen3"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--four-bit", action="store_true")
    ap.add_argument(
        "--max-chunks-per-doc",
        type=int,
        default=4,
        help="cap chunks scored per candidate doc (max-pooled); 0 = all",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pool = json.loads(POOL.read_text())
    qrels = json.loads(QRELS.read_text())
    threshold = int(qrels.get("relevance_threshold", DEFAULT_RELEVANCE_THRESHOLD))
    labels = graded_labels(qrels)
    cases = pool["cases"]

    print(f"[{args.tag}] loading chunk text from {POOL_PROFILE} lancedb ...", flush=True)
    chunks_by_doc = load_chunks_by_doc(POOL_PROFILE)
    print(f"[{args.tag}] docs with chunks: {len(chunks_by_doc)}", flush=True)

    t0 = time.time()
    backend = _bk.make_backend(args)
    print(f"[{args.tag}] backend loaded in {time.time()-t0:.1f}s", flush=True)

    per_case, t_nd, h_nd = [], [], []
    missing_docs = 0
    for i, (cid, case) in enumerate(sorted(cases.items())):
        query = case["query"]
        pooled = case["pooled"]
        rel = set(
            relevant_from_grades(labels.get(cid, {}), threshold=threshold)
        ) | set(case.get("curated_relevant") or [])
        if not rel:
            continue
        doc_ids = list(pooled.keys())
        # flatten all chunks of all candidate docs, remember which doc each belongs to
        flat_docs, flat_owner = [], []
        for did in doc_ids:
            chs = chunks_by_doc.get(did)
            if not chs:
                # fall back to pool excerpt if doc not in this index
                missing_docs += 1
                chs = [_bk.doc_text(pooled[did])]
            if args.max_chunks_per_doc and len(chs) > args.max_chunks_per_doc:
                # longest chunks first as a cheap proxy for content-bearing
                chs = sorted(chs, key=len, reverse=True)[: args.max_chunks_per_doc]
            for c in chs:
                flat_docs.append(c)
                flat_owner.append(did)
        scr = backend.score(query, flat_docs, args.batch_size)
        # max-pool per doc
        best = {}
        for owner, s in zip(flat_owner, scr):
            if owner not in best or s > best[owner]:
                best[owner] = s
        order = sorted(doc_ids, key=lambda d: -best.get(d, -1e9))
        tm = case_metrics(_bk.rows_from_order(order, pooled), rel)
        hm = case_metrics(_bk.rows_from_order(_bk.hybrid_order(pooled), pooled), rel)
        t_nd.append(tm["ndcg@10"])
        h_nd.append(hm["ndcg@10"])
        per_case.append(
            {
                "case_id": cid,
                "teacher_ndcg@10": tm["ndcg@10"],
                "hybrid_ndcg@10": hm["ndcg@10"],
                "n_chunks_scored": len(flat_docs),
            }
        )
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(
            f"[{args.tag}] [{i+1}/{len(cases)}] {cid:42s} "
            f"t={tm['ndcg@10']:.4f} h={hm['ndcg@10']:.4f} "
            f"chunks={len(flat_docs)} gpu={mem:.2f}GB",
            flush=True,
        )

    n = len(t_nd)
    summary = {
        "tag": args.tag,
        "model": args.model,
        "backend": args.backend,
        "four_bit": args.four_bit,
        "text_field": "best_chunk_fulltext(max_pool)",
        "max_chunks_per_doc": args.max_chunks_per_doc,
        "pool_profile": POOL_PROFILE,
        "n_cases": n,
        "docs_missing_from_index_fallback_to_excerpt": missing_docs,
        "teacher_mean_ndcg@10": sum(t_nd) / n,
        "hybrid_mean_ndcg@10": sum(h_nd) / n,
        "delta_vs_hybrid": (sum(t_nd) - sum(h_nd)) / n,
        "wins": sum(1 for c in per_case if c["teacher_ndcg@10"] > c["hybrid_ndcg@10"] + 1e-9),
        "losses": sum(1 for c in per_case if c["hybrid_ndcg@10"] > c["teacher_ndcg@10"] + 1e-9),
        "ties": sum(1 for c in per_case if abs(c["teacher_ndcg@10"] - c["hybrid_ndcg@10"]) <= 1e-9),
        "gpu_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
        "gpu_device": torch.cuda.get_device_name(0),
    }
    (OUT_DIR / f"{args.tag}.json").write_text(
        json.dumps({"summary": summary, "per_case": per_case}, indent=2)
    )
    print(f"\n=== SUMMARY [{args.tag}] ===\n{json.dumps(summary, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
