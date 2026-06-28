"""GATE 1 — non-circular held-out-positive RANK test (stock vs fine-tuned SPLADE).

For each HELD-OUT query (unseen in training), build a FIXED shared candidate pool:
    pool = {true positive chunk} ∪ {its mined hard negatives} ∪ {fixed random corpus sample}
Encode query + pool with (a) the STOCK opensearch-distill base and (b) the fine-tuned
merged SPLADE-v1. Rank by sparse dot-product. Metric = RANK POSITION of the true
positive → MRR + recall@{1,5,10}, stock vs fine-tuned. This is NON-CIRCULAR: it scores
rank of the real seed positive on a fixed shared pool, NOT agreement with any teacher.

Runs in the TRAINING env (needs SparseEncoder). $0, single GPU.

Inputs:
  --heldout   data/training/gate1/triples_heldout.jsonl  (queries NOT seen in training)
  --train     data/training/gate1/triples_train.jsonl    (sanity: should improve MORE)
  --distractors data/training/gate1/distractor_pool.jsonl  (fixed random corpus sample:
                {chunk_id, text}) — SAME pool for every query and both models.
  --base      opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill
  --finetuned data/training/checkpoints/splade-gd-v1   (merged)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _sparse_vec(tensor) -> tuple[np.ndarray, np.ndarray]:
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"expected sparse torch.Tensor, got {type(tensor)!r}")
    c = tensor.coalesce()
    idx = c.indices()[0] if c.dim() == 1 else c.indices()[1]
    return idx.cpu().numpy().astype(np.int64), c.values().cpu().numpy().astype(np.float32)


def _dot(qi, qv, di, dv) -> float:
    """Sparse dot product of two (indices, values) vectors."""
    if len(qi) == 0 or len(di) == 0:
        return 0.0
    qmap = dict(zip(qi.tolist(), qv.tolist()))
    s = 0.0
    for i, v in zip(di.tolist(), dv.tolist()):
        w = qmap.get(i)
        if w is not None:
            s += w * v
    return s


def _encode_model(model_name: str, device: str):
    from sentence_transformers import SparseEncoder

    return SparseEncoder(model_name, device=device)


def _encode_docs(model, texts: list[str], batch_size: int) -> list[tuple[np.ndarray, np.ndarray]]:
    embs = model.encode_document(
        texts, batch_size=batch_size, convert_to_tensor=False,
        convert_to_sparse_tensor=True, show_progress_bar=False,
    )
    return [_sparse_vec(t) for t in embs]


def _encode_queries(model, texts: list[str], batch_size: int) -> list[tuple[np.ndarray, np.ndarray]]:
    embs = model.encode_query(
        texts, batch_size=batch_size, convert_to_tensor=False,
        convert_to_sparse_tensor=True, show_progress_bar=False,
    )
    return [_sparse_vec(t) for t in embs]


def _build_pools(rows: list[dict], distractors: list[dict], max_hard: int):
    """For each query, build the candidate pool. Returns:
      pools: list of (pool_texts, pool_ids, positive_index)
    Distractor chunk_ids overlapping the positive/hard-negs are skipped so the
    positive index is unambiguous and never double-counted.
    """
    dist_texts = [d["text"] for d in distractors]
    dist_ids = [d["chunk_id"] for d in distractors]
    pools = []
    for r in rows:
        pos_id = r["positive_chunk_id"]
        pos_text = r["positive_text"]
        neg_ids = (r.get("negative_chunk_ids") or [])[:max_hard]
        neg_texts = (r.get("negative_texts") or [])[:max_hard]
        used = {pos_id, *neg_ids}
        texts = [pos_text] + list(neg_texts)
        ids = [pos_id] + list(neg_ids)
        for did, dt in zip(dist_ids, dist_texts):
            if did in used:
                continue
            ids.append(did)
            texts.append(dt)
        pools.append((texts, ids, 0))  # positive is always index 0
    return pools


def _rank_of_positive(qvec, doc_vecs, pos_idx: int) -> int:
    """1-based rank of the positive among the pool (higher dot = better)."""
    qi, qv = qvec
    scores = np.array([_dot(qi, qv, di, dv) for (di, dv) in doc_vecs], dtype="float64")
    # rank = 1 + number of docs scoring strictly higher than the positive.
    pos_score = scores[pos_idx]
    higher = int(np.sum(scores > pos_score))
    return higher + 1


def _eval_model(model_name: str, device: str, pools, queries: list[str], batch_size: int) -> dict:
    model = _encode_model(model_name, device)
    qvecs = _encode_queries(model, queries, batch_size)
    ranks = []
    for (texts, ids, pos_idx), qvec in zip(pools, qvecs):
        doc_vecs = _encode_docs(model, texts, batch_size)
        ranks.append(_rank_of_positive(qvec, doc_vecs, pos_idx))
    ranks = np.array(ranks, dtype="int64")
    # free GPU between models
    del model
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return {
        "n": int(len(ranks)),
        "mrr": float(np.mean(1.0 / ranks)),
        "recall@1": float(np.mean(ranks <= 1)),
        "recall@5": float(np.mean(ranks <= 5)),
        "recall@10": float(np.mean(ranks <= 10)),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "ranks": ranks.tolist(),
    }


def _report(name: str, stock: dict, ft: dict) -> dict:
    out = {"split": name, "n": stock["n"], "stock": {}, "finetuned": {}, "delta": {}}
    for k in ("mrr", "recall@1", "recall@5", "recall@10", "mean_rank", "median_rank"):
        out["stock"][k] = round(stock[k], 4)
        out["finetuned"][k] = round(ft[k], 4)
        out["delta"][k] = round(ft[k] - stock[k], 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--train", default="")
    ap.add_argument("--distractors", required=True)
    ap.add_argument("--base", default="opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill")
    ap.add_argument("--finetuned", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-hard", type=int, default=7, help="max hard negs per query in pool")
    ap.add_argument("--train-sanity-n", type=int, default=40, help="subset of train queries for sanity")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    heldout = _load_jsonl(Path(args.heldout))
    distractors = _load_jsonl(Path(args.distractors))
    print(f"held-out queries: {len(heldout)}  distractors: {len(distractors)}", file=sys.stderr)

    ho_pools = _build_pools(heldout, distractors, args.max_hard)
    ho_queries = [r["query"] for r in heldout]
    pool_size = len(ho_pools[0][0]) if ho_pools else 0
    print(f"pool size (1 pos + {args.max_hard} hard + distractors): {pool_size}", file=sys.stderr)

    print("=== encoding held-out with STOCK base ===", file=sys.stderr)
    ho_stock = _eval_model(args.base, args.device, ho_pools, ho_queries, args.batch_size)
    print("=== encoding held-out with FINE-TUNED ===", file=sys.stderr)
    ho_ft = _eval_model(args.finetuned, args.device, ho_pools, ho_queries, args.batch_size)

    results = {"pool_size": pool_size, "max_hard": args.max_hard,
               "base": args.base, "finetuned": args.finetuned,
               "heldout": _report("heldout", ho_stock, ho_ft)}

    if args.train:
        tr = _load_jsonl(Path(args.train))[: args.train_sanity_n]
        tr_pools = _build_pools(tr, distractors, args.max_hard)
        tr_queries = [r["query"] for r in tr]
        print("=== encoding train-sanity with STOCK base ===", file=sys.stderr)
        tr_stock = _eval_model(args.base, args.device, tr_pools, tr_queries, args.batch_size)
        print("=== encoding train-sanity with FINE-TUNED ===", file=sys.stderr)
        tr_ft = _eval_model(args.finetuned, args.device, tr_pools, tr_queries, args.batch_size)
        results["train_sanity"] = _report("train_sanity", tr_stock, tr_ft)

    # strip per-query ranks from the printed summary (keep in --out file)
    summary = json.loads(json.dumps(results))
    print("GATE1_RANK_RESULT:", json.dumps(summary))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
