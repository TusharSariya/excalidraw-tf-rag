"""Build contrastive training triples from training queries (no teacher margins).

For each training query (positive = its source chunk):
  1. MINE negatives = BM25 top-K ∪ dense top-K over the production index,
     MINUS the positive chunk AND all sibling chunks (same source_doc_id).
  2. FALSE-NEGATIVE DENOISE with Qwen3-Reranker-4B (nf4 4-bit, max_length 512,
     small batch — reusing teacher_bakeoff's OOM-safe config): drop any mined
     negative the reranker scores within ``--denoise-margin`` of the POSITIVE's
     own reranker score (likely a true positive mislabeled as a negative).
  3. Keep ~``--keep-negs`` negatives. Emit inline chunk TEXT (contrastive MNRL
     needs text, not teacher scores).

Two GPU models never co-resident in 8 GB:
  - ``--phase mine``    : load the dense embedder (Qwen3-Embedding-4B) → candidates.jsonl
  - ``--phase denoise`` : load Qwen3-Reranker-4B → triples.jsonl (dense model freed)
  - ``--phase all``     : run mine then denoise sequentially (default; frees between).

Run on the desktop GPU box:
    uv run --no-sync python build_triples.py --phase all --embed-profile cuda-qwen4b-1024
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_triples")

TOOL_ROOT = Path(__file__).resolve().parent.parent
SRC = TOOL_ROOT / "src"
# Data dir is env-overridable so a separate run (e.g. GATE-1) can use its own
# namespace of artifacts without clobbering the default data/training/ files.
TRAIN_DATA_DIR = Path(os.getenv("GRAPH_RAG_TRAIN_DATA_DIR", str(TOOL_ROOT / "data" / "training")))
QUERIES_PATH = TRAIN_DATA_DIR / "queries.jsonl"
CANDIDATES_PATH = TRAIN_DATA_DIR / "candidates.jsonl"
TRIPLES_PATH = TRAIN_DATA_DIR / "triples.jsonl"

_RAG_COMMON_SRC = TOOL_ROOT.parent / "rag-common" / "src"
for _p in (SRC, _RAG_COMMON_SRC):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #
def filter_negatives(
    candidates: list[tuple[str, str]],
    *,
    positive_id: str,
    positive_doc: str | None,
) -> list[tuple[str, str]]:
    """Drop the positive chunk and every sibling chunk (same source_doc_id).

    ``candidates`` = list of (chunk_id, doc_id). Order preserved.
    """
    out: list[tuple[str, str]] = []
    for cid, cdoc in candidates:
        if cid == positive_id:
            continue
        if positive_doc is not None and cdoc == positive_doc:
            continue  # sibling chunk -> not a true negative
        out.append((cid, cdoc))
    return out


def denoise_negatives(
    scored: list[tuple[str, float]],
    *,
    positive_score: float,
    margin: float,
) -> tuple[list[tuple[str, float]], int]:
    """Drop negatives scored within ``margin`` of the positive's reranker score.

    A mined "negative" the reranker likes almost as much as the true positive is
    probably itself relevant (false negative); keeping it would push the model to
    separate two relevant docs. Returns (kept, n_dropped).
    """
    kept: list[tuple[str, float]] = []
    dropped = 0
    for cid, score in scored:
        if score >= positive_score - margin:
            dropped += 1
            continue
        kept.append((cid, score))
    return kept, dropped


# --------------------------------------------------------------------------- #
# Corpus / mining
# --------------------------------------------------------------------------- #
def _load_chunk_index() -> dict[str, dict]:
    """chunk_id -> {doc_id, text} for inlining + sibling lookup."""
    import lancedb

    from graph_layout_rag.paths import CHUNKS_TABLE, profile_index_paths

    profile = os.getenv("GRAPH_RAG_TRAIN_CORPUS_PROFILE", "cuda-qwen0.6b-1024")
    paths = profile_index_paths(profile)
    db = lancedb.connect(str(paths.lance_dir))
    rows = db.open_table(CHUNKS_TABLE).to_arrow().to_pylist()
    return {
        r["id"]: {"doc_id": r.get("doc_id"), "text": (r.get("text") or "")}
        for r in rows
    }


def _read_queries(limit: int = 0) -> list[dict]:
    rows = [json.loads(line) for line in QUERIES_PATH.read_text().splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def phase_mine(args) -> None:
    """BM25 ∪ dense candidate mining → candidates.jsonl (one row per query)."""
    from graph_layout_rag.query.retrieve import retrieve_candidates

    chunk_index = _load_chunk_index()
    queries = _read_queries(limit=args.max_queries)
    log.info("mining negatives for %d queries (profile=%s)", len(queries), args.embed_profile)

    n_written = 0
    with CANDIDATES_PATH.open("w") as out:
        for i, rec in enumerate(queries, 1):
            q = rec["query"]
            pos_id = rec["source_chunk_id"]
            pos_doc = rec.get("source_doc_id")

            try:
                dense = retrieve_candidates(
                    q, top=args.mine_k, embed_profile=args.embed_profile,
                    hybrid=False, pool=args.mine_k,
                )
                sparse = retrieve_candidates(
                    q, top=args.mine_k, embed_profile=args.embed_profile,
                    sparse_only=True, pool=args.mine_k,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("mining failed for %s: %s", rec["id"], exc)
                continue

            # Union, preserve a stable order (dense first, then sparse-only extras).
            seen: set[str] = set()
            cand: list[tuple[str, str]] = []
            for row in list(dense) + list(sparse):
                cid = row.get("id")
                if not cid or cid in seen:
                    continue
                seen.add(cid)
                cand.append((cid, row.get("doc_id")))

            negs = filter_negatives(cand, positive_id=pos_id, positive_doc=pos_doc)
            if not negs:
                continue
            if pos_id not in chunk_index:
                continue

            out.write(json.dumps({
                "id": rec["id"],
                "query": q,
                "positive_chunk_id": pos_id,
                "positive_doc_id": pos_doc,
                "negative_chunk_ids": [c[0] for c in negs[: args.mine_pool]],
            }) + "\n")
            n_written += 1
            if i % 500 == 0:
                log.info("mined %d/%d (%d written)", i, len(queries), n_written)
    log.info("wrote %d candidate rows -> %s", n_written, CANDIDATES_PATH)


# --------------------------------------------------------------------------- #
# Denoise (Qwen3-Reranker-4B, nf4 4-bit) — reuse teacher_bakeoff config
# --------------------------------------------------------------------------- #
class Qwen3Reranker:
    """P(yes) reranker, nf4 4-bit, max_length 512 (OOM-safe for 8 GB)."""

    PREFIX = (
        "<|im_start|>system\nJudge whether the Document meets the requirements "
        "based on the Query and the Instruct provided. Note that the answer can "
        'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    INSTRUCT = (
        "Given a graph-drawing / graph-layout search query, retrieve passages "
        "that are relevant to answering it."
    )

    def __init__(self, model: str, max_length: int = 512):
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                   BitsAndBytesConfig)

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            ),
            device_map="cuda",
        ).eval()
        self.max_length = max_length
        self.tok_true = self.tok.convert_tokens_to_ids("yes")
        self.tok_false = self.tok.convert_tokens_to_ids("no")
        self.pre_ids = self.tok.encode(self.PREFIX, add_special_tokens=False)
        self.suf_ids = self.tok.encode(self.SUFFIX, add_special_tokens=False)

    def _fmt(self, query: str, doc: str) -> str:
        return f"<Instruct>: {self.INSTRUCT}\n<Query>: {query}\n<Document>: {doc}"

    def score(self, query: str, docs: list[str], bs: int = 8) -> list[float]:
        import torch.nn.functional as F

        torch = self.torch
        scores: list[float] = []
        body_max = self.max_length - len(self.pre_ids) - len(self.suf_ids)
        with torch.no_grad():
            for i in range(0, len(docs), bs):
                batch = docs[i : i + bs]
                texts = [self._fmt(query, d) for d in batch]
                enc = self.tok(texts, truncation=True, max_length=body_max,
                               add_special_tokens=False)
                input_ids = [self.pre_ids + ids + self.suf_ids for ids in enc["input_ids"]]
                pad = self.tok.pad({"input_ids": input_ids}, padding=True,
                                   return_tensors="pt").to("cuda")
                logits = self.model(**pad).logits[:, -1, :]
                tf = logits[:, [self.tok_false, self.tok_true]]
                p_yes = F.log_softmax(tf, dim=1)[:, 1].exp()
                scores.extend(p_yes.float().cpu().tolist())
        return scores


def phase_denoise(args) -> None:
    chunk_index = _load_chunk_index()
    cand_rows = [json.loads(l) for l in CANDIDATES_PATH.read_text().splitlines() if l.strip()]
    if args.max_candidates:
        cand_rows = cand_rows[: args.max_candidates]
    if args.neg_cap:
        # Cap candidates scored per query to bound reranker cost (keep the first
        # N mined negatives; mining already ordered dense-first then sparse extras).
        for r in cand_rows:
            r["negative_chunk_ids"] = r["negative_chunk_ids"][: args.neg_cap]
    log.info("denoising %d candidate rows with %s", len(cand_rows), args.reranker)

    reranker = Qwen3Reranker(args.reranker, max_length=args.max_length)

    n_written = 0
    total_dropped = 0
    with TRIPLES_PATH.open("w") as out:
        for i, row in enumerate(cand_rows, 1):
            q = row["query"]
            pos_id = row["positive_chunk_id"]
            pos = chunk_index.get(pos_id)
            if pos is None:
                continue
            pos_text = pos["text"][: args.text_chars]

            neg_ids = [c for c in row["negative_chunk_ids"] if c in chunk_index]
            if not neg_ids:
                continue
            neg_texts = [chunk_index[c]["text"][: args.text_chars] for c in neg_ids]

            # Score positive + all negatives in one query context.
            all_scores = reranker.score(q, [pos_text] + neg_texts, bs=args.batch_size)
            pos_score = all_scores[0]
            neg_scored = list(zip(neg_ids, all_scores[1:]))

            kept, dropped = denoise_negatives(
                neg_scored, positive_score=pos_score, margin=args.denoise_margin
            )
            total_dropped += dropped
            # Hardest first (highest score among the kept = closest to positive).
            kept.sort(key=lambda x: x[1], reverse=True)
            kept = kept[: args.keep_negs]
            if len(kept) < args.min_negs:
                continue

            out.write(json.dumps({
                "id": row["id"],
                "query": q,
                "positive_text": pos_text,
                "negative_texts": [chunk_index[c]["text"][: args.text_chars] for c, _ in kept],
                "positive_chunk_id": pos_id,
                "negative_chunk_ids": [c for c, _ in kept],
                "positive_reranker_score": round(pos_score, 4),
            }) + "\n")
            n_written += 1
            if i % 200 == 0:
                log.info("denoised %d/%d (%d written, %d dropped)",
                         i, len(cand_rows), n_written, total_dropped)

    summary = {
        "triples": n_written,
        "denoise_dropped": total_dropped,
        "denoise_margin": args.denoise_margin,
        "keep_negs": args.keep_negs,
        "reranker": args.reranker,
        "avg_negs": None,
    }
    if n_written:
        negs = [len(json.loads(l)["negative_texts"])
                for l in TRIPLES_PATH.read_text().splitlines() if l.strip()]
        summary["avg_negs"] = round(sum(negs) / len(negs), 2)
    (TRAIN_DATA_DIR / "triples_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("wrote %d triples -> %s", n_written, TRIPLES_PATH)
    print("TRIPLES_SUMMARY:", json.dumps(summary))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["mine", "denoise", "all"], default="all")
    ap.add_argument("--embed-profile", default="cuda-qwen4b-1024")
    ap.add_argument("--max-queries", type=int, default=0,
                    help="cap queries processed (0 = all; Step-1 uses a subset to "
                         "validate the pipeline — full mining is GATE-1's job)")
    ap.add_argument("--mine-k", type=int, default=50, help="BM25 / dense top-K each")
    ap.add_argument("--mine-pool", type=int, default=30,
                    help="max candidate negatives carried into denoise")
    ap.add_argument("--reranker", default="Qwen/Qwen3-Reranker-4B")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--denoise-margin", type=float, default=0.10)
    ap.add_argument("--keep-negs", type=int, default=7)
    ap.add_argument("--min-negs", type=int, default=2)
    ap.add_argument("--max-candidates", type=int, default=0,
                    help="cap candidate ROWS denoised (0 = all)")
    ap.add_argument("--neg-cap", type=int, default=0,
                    help="cap mined negatives SCORED per query (0 = all ~25) to bound "
                         "reranker cost")
    ap.add_argument("--text-chars", type=int, default=1600)
    args = ap.parse_args()

    TRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("GRAPH_RAG_TRAIN_CORPUS_PROFILE", args.embed_profile)

    if args.phase in ("mine", "all"):
        phase_mine(args)
    if args.phase in ("denoise", "all"):
        phase_denoise(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
