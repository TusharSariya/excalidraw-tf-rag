#!/usr/bin/env python3
"""GATE 0 follow-up — teacher bake-off across rerankers on the catalog pool.

Same 49 catalog cases, same same-pool weighted-hybrid baseline, same binary
case_metrics nDCG@10 as scripts/teacher_sanity_rerank.py.

Reranks each case's already-pooled candidates (short pool excerpt) with one
reranker chosen by --model, on the desktop GPU. Fail-loud on OOM (no CPU
fallback). Writes per-model JSON under data/eval/teacher_sanity/bakeoff/.

Backends:
  qwen3      Qwen/Qwen3-Reranker-{0.6B,4B}  (causal-LM yes/no scoring; 4-bit opt)
  ce         sentence-transformers CrossEncoder (gte-modernbert etc.)
  mxbai      mxbai-rerank MxbaiRerankV2

Usage (desktop):
  uv run --with ... python scripts/teacher_bakeoff.py --backend qwen3 \
      --model Qwen/Qwen3-Reranker-0.6B --tag qwen3-0.6b
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

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "data" / "eval" / "pool" / "catalog" / "pool.json"
QRELS = ROOT / "data" / "eval" / "qrels" / "catalog" / "qrels.json"
OUT_DIR = ROOT / "data" / "eval" / "teacher_sanity" / "bakeoff"
HYBRID_SYSTEM = "hybrid"


def doc_text(meta: dict) -> str:
    parts = []
    if meta.get("title"):
        parts.append(str(meta["title"]))
    if meta.get("abstract"):
        parts.append(str(meta["abstract"]))
    if meta.get("excerpt"):
        parts.append(str(meta["excerpt"]))
    return "\n".join(parts).strip() or str(meta.get("doc_id") or "")


def rows_from_order(order, pooled):
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


def hybrid_order(pooled):
    BIG = 10**9
    return [
        d
        for d, _ in sorted(
            pooled.items(),
            key=lambda kv: kv[1].get("systems", {}).get(HYBRID_SYSTEM, BIG),
        )
    ]


# ---------------- backends: score(query, docs) -> list[float] ----------------
class CEBackend:
    def __init__(self, model, max_length):
        from sentence_transformers import CrossEncoder

        self.m = CrossEncoder(
            model, device="cuda", max_length=max_length,
            automodel_args={"torch_dtype": torch.float16},
        )

    def score(self, query, docs, bs):
        import numpy as np

        scores = self.m.predict(
            [[query, d] for d in docs], batch_size=bs, show_progress_bar=False
        )
        return np.asarray(scores, dtype=float).tolist()


class MxbaiBackend:
    def __init__(self, model, max_length):
        from mxbai_rerank import MxbaiRerankV2

        self.m = MxbaiRerankV2(
            model, device="cuda", torch_dtype=torch.float16, max_length=max_length
        )

    def score(self, query, docs, bs):
        ranked = self.m.rank(
            query, docs, top_k=len(docs), batch_size=bs, return_documents=False,
            sort=False,
        )
        # sort=False -> results in input order; .score per item
        out = [0.0] * len(docs)
        for r in ranked:
            out[r.index] = float(r.score)
        return out


class Qwen3Backend:
    """Qwen3-Reranker: causal LM, P(yes) over a yes/no judgement prompt."""

    PREFIX = (
        "<|im_start|>system\nJudge whether the Document meets the requirements "
        "based on the Query and the Instruct provided. Note that the answer can "
        'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    SUFFIX = (
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    INSTRUCT = (
        "Given a graph-drawing / graph-layout search query, retrieve passages "
        "that are relevant to answering it."
    )

    def __init__(self, model, max_length, four_bit):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        kw = dict(torch_dtype=torch.float16)
        if four_bit:
            from transformers import BitsAndBytesConfig

            kw = dict(
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                ),
                device_map="cuda",
            )
        self.model = AutoModelForCausalLM.from_pretrained(model, **kw).eval()
        if not four_bit:
            self.model = self.model.to("cuda")
        self.max_length = max_length
        self.tok_true = self.tok.convert_tokens_to_ids("yes")
        self.tok_false = self.tok.convert_tokens_to_ids("no")
        self.pre_ids = self.tok.encode(self.PREFIX, add_special_tokens=False)
        self.suf_ids = self.tok.encode(self.SUFFIX, add_special_tokens=False)

    def _fmt(self, query, doc):
        return (
            f"<Instruct>: {self.INSTRUCT}\n<Query>: {query}\n<Document>: {doc}"
        )

    @torch.no_grad()
    def score(self, query, docs, bs):
        import torch.nn.functional as F

        scores = []
        body_max = self.max_length - len(self.pre_ids) - len(self.suf_ids)
        for i in range(0, len(docs), bs):
            batch = docs[i : i + bs]
            texts = [self._fmt(query, d) for d in batch]
            enc = self.tok(
                texts, truncation=True, max_length=body_max,
                add_special_tokens=False,
            )
            input_ids = [self.pre_ids + ids + self.suf_ids for ids in enc["input_ids"]]
            pad = self.tok.pad(
                {"input_ids": input_ids}, padding=True, return_tensors="pt"
            ).to("cuda")
            logits = self.model(**pad).logits[:, -1, :]
            tf = logits[:, [self.tok_false, self.tok_true]]
            p_yes = F.log_softmax(tf, dim=1)[:, 1].exp()
            scores.extend(p_yes.float().cpu().tolist())
        return scores


def make_backend(args):
    if args.backend == "ce":
        return CEBackend(args.model, args.max_length)
    if args.backend == "mxbai":
        return MxbaiBackend(args.model, args.max_length)
    if args.backend == "qwen3":
        return Qwen3Backend(args.model, args.max_length, args.four_bit)
    raise SystemExit(f"unknown backend {args.backend}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["ce", "mxbai", "qwen3"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--four-bit", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pool = json.loads(POOL.read_text())
    qrels = json.loads(QRELS.read_text())
    threshold = int(qrels.get("relevance_threshold", DEFAULT_RELEVANCE_THRESHOLD))
    labels = graded_labels(qrels)
    cases = pool["cases"]
    print(
        f"[{args.tag}] backend={args.backend} model={args.model} "
        f"max_len={args.max_length} bs={args.batch_size} 4bit={args.four_bit} "
        f"cases={len(cases)} thr>={threshold}",
        flush=True,
    )

    t0 = time.time()
    backend = make_backend(args)
    print(f"[{args.tag}] loaded in {time.time()-t0:.1f}s", flush=True)

    per_case, t_nd, h_nd = [], [], []
    for i, (cid, case) in enumerate(sorted(cases.items())):
        query = case["query"]
        pooled = case["pooled"]
        rel = set(
            relevant_from_grades(labels.get(cid, {}), threshold=threshold)
        ) | set(case.get("curated_relevant") or [])
        if not rel:
            continue
        doc_ids = list(pooled.keys())
        docs = [doc_text(pooled[d]) for d in doc_ids]
        scr = backend.score(query, docs, args.batch_size)
        order = [doc_ids[j] for j in sorted(range(len(docs)), key=lambda j: -scr[j])]
        tm = case_metrics(rows_from_order(order, pooled), rel)
        hm = case_metrics(rows_from_order(hybrid_order(pooled), pooled), rel)
        t_nd.append(tm["ndcg@10"])
        h_nd.append(hm["ndcg@10"])
        per_case.append(
            {
                "case_id": cid,
                "category": case.get("category"),
                "teacher_ndcg@10": tm["ndcg@10"],
                "hybrid_ndcg@10": hm["ndcg@10"],
                "n_relevant": len(rel),
            }
        )
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(
            f"[{args.tag}] [{i+1}/{len(cases)}] {cid:46s} "
            f"t={tm['ndcg@10']:.4f} h={hm['ndcg@10']:.4f} gpu={mem:.2f}GB",
            flush=True,
        )

    n = len(t_nd)
    summary = {
        "tag": args.tag,
        "model": args.model,
        "backend": args.backend,
        "max_length": args.max_length,
        "four_bit": args.four_bit,
        "text_field": "pool_excerpt",
        "n_cases": n,
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
