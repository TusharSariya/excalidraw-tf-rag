"""Generate label-free SPLADE training queries over LanceDB chunks ($0, local LLM).

Each training query is grounded in ONE source chunk; that chunk IS the positive
for contrastive training (no teacher, no cloud LLM). Generation uses a LOCAL model
(Ollama gemma4:e4b on the desktop GPU) — never a cloud API (the gemini judge is
reserved for GATE 3). Output namespace: ``train-v1:``.

HARD DEDUP GATE (training-only; gold_synth's eval-facing filters are NOT touched):
drop any training query whose token-Jaccard > ``JACCARD_MAX`` (0.45) OR whose
embedding-cosine > ``COSINE_MAX`` (0.85) against ANY of the 175 eval queries.
After filtering we ASSERT zero survivors above either threshold and log per-gate
drop counts. This is the contamination firewall between training data and eval.

Run on the desktop GPU box (Ollama local + a local embedder):
    uv run --no-sync python gen_train_queries.py --target 12000 --concurrency 8
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gen_train_queries")

# --- Dedup thresholds (training-only firewall) -------------------------------
JACCARD_MAX = 0.45
COSINE_MAX = 0.85
# Canonical thresholds for the post-filter SAFETY assertion. Snapshotted as
# separate constants so that even if the (patchable) filter gates above are ever
# loosened/bypassed, the assertion still catches a contaminating survivor at the
# real contamination boundary. The invariant the campaign depends on is exactly:
# "no shipped training query exceeds 0.45 Jaccard or 0.85 cosine vs any eval query".
ASSERT_JACCARD_MAX = 0.45
ASSERT_COSINE_MAX = 0.85
NAMESPACE = os.getenv("GRAPH_RAG_TRAIN_NAMESPACE", "train-v1:")

# --- Paths (resolved relative to the eval package so we share the corpus) -----
TOOL_ROOT = Path(__file__).resolve().parent.parent  # tools/graph-layout-rag
SRC = TOOL_ROOT / "src"
# Data dir is env-overridable so a separate run (e.g. GATE-1) can use its own
# namespace of artifacts without clobbering the default data/training/ files.
TRAIN_DATA_DIR = Path(os.getenv("GRAPH_RAG_TRAIN_DATA_DIR", str(TOOL_ROOT / "data" / "training")))
QUERIES_PATH = TRAIN_DATA_DIR / "queries.jsonl"

_RAG_COMMON_SRC = TOOL_ROOT.parent / "rag-common" / "src"
for _p in (SRC, _RAG_COMMON_SRC):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# NOTE: graph_layout_rag (and its rag_common dependency) are imported lazily inside
# the functions that need the corpus, so the pure dedup helpers stay unit-testable
# in the isolated training env without those packages installed.

CORPUS_PROFILE = os.getenv("GRAPH_RAG_TRAIN_CORPUS_PROFILE", "cuda-qwen0.6b-1024")
OLLAMA_HOST = os.getenv("RAG_OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("RAG_TRAIN_GEN_MODEL", "gemma4:e4b-it-qat")

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "has", "have", "can", "via", "use", "used", "using", "based", "into", "such",
    "which", "their", "between", "also", "these", "those", "than", "then", "but",
    "not", "all", "any", "how", "what", "when", "where", "why", "does",
}


def _tokens(text: str) -> set[str]:
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {t for t in toks if t not in _STOPWORDS and len(t) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# --- Eval query strings (the 175-case contamination set) ----------------------
def load_eval_queries() -> list[str]:
    """All eval query strings = curated GOLD_CASES ∪ synthetic gold cases.

    These are exactly the strings behind the 49 catalog + 175 catalog-nl qrels
    cases. We dedup training queries against every one of them.
    """
    os.environ.setdefault("GRAPH_RAG_INCLUDE_SYNTH", "1")
    from graph_layout_rag.eval.gold_cases import GOLD_CASES
    from graph_layout_rag.eval.gold_synth import load_synth_cases

    queries = [c.query for c in GOLD_CASES]
    try:
        queries += [r["query"] for r in load_synth_cases() if r.get("query")]
    except Exception as exc:  # noqa: BLE001
        log.warning("could not load synth cases (%s); using curated only", exc)
    # de-dup identical strings but keep order
    seen, out = set(), []
    for q in queries:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


# --- Local embedder for cosine dedup -----------------------------------------
def _build_embedder():
    """Small local sentence embedder for cosine-leak detection.

    We use a CPU MiniLM (downloaded once, ~90MB) so it never contends with the
    GPU SPLADE/reranker jobs and stays $0. Cosine here is only a *near-duplicate*
    gate, so exact parity with the production dense model is unnecessary.
    """
    from sentence_transformers import SentenceTransformer

    name = os.getenv("RAG_TRAIN_DEDUP_EMBED", "sentence-transformers/all-MiniLM-L6-v2")
    return SentenceTransformer(name, device="cpu")


def _embed(embedder, texts: list[str]):
    import numpy as np

    if not texts:
        return np.zeros((0, 384), dtype="float32")
    vecs = embedder.encode(
        texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vecs.astype("float32")


# --- Chunk sampling -----------------------------------------------------------
def load_chunks() -> list[dict]:
    import lancedb

    from graph_layout_rag.paths import CHUNKS_TABLE, profile_index_paths

    paths = profile_index_paths(CORPUS_PROFILE)
    db = lancedb.connect(str(paths.lance_dir))
    rows = db.open_table(CHUNKS_TABLE).to_arrow().to_pylist()
    out = []
    for r in rows:
        text = (r.get("text") or "").strip()
        if len(text) < 200:  # skip stubs/images-only chunks
            continue
        out.append({"id": r["id"], "doc_id": r.get("doc_id"),
                    "title": r.get("title") or "", "text": text})
    return out


# --- Prompting (diverse query types to fight extractive skew) ------------------
PROMPT_STYLES = [
    ("keyword", "Write ONE short keyword search query (3-8 words, no punctuation) that a "
                "researcher would type to FIND the passage below. Use the passage's "
                "technical vocabulary. Output ONLY the query."),
    ("question", "Write ONE natural-language question a researcher would ask whose answer "
                 "is found in the passage below. Do NOT copy sentences verbatim. Output "
                 "ONLY the question."),
    ("problem", "State ONE practical problem (one sentence, problem-first, indirect "
                "phrasing) whose solution is described in the passage below. Avoid reusing "
                "the passage's exact wording. Output ONLY the problem statement."),
]


def _build_prompt(chunk: dict, style_instr: str) -> str:
    body = chunk["text"][:1200]
    title = chunk["title"][:160]
    return (
        f"{style_instr}\n\n"
        f"Title: {title}\nPassage:\n{body}\n\nQuery:"
    )


# --- Extractive generation (instant, $0, no GPU) ------------------------------
# The local LLM (gemma3-QAT) emits ~450-540 hidden tokens per query (~0.3-0.7 q/s),
# which is too slow for a meaningful batch. The plan sanctions "a LOCAL LLM ... OR an
# extractive method"; extractive is the practical $0 path at scale. We build a query
# from the chunk's title + its most salient content terms (and section-path terms),
# which yields a realistic (query -> source-chunk) positive. The HARD DEDUP GATE then
# guarantees no eval contamination regardless of how queries are produced.
_CONTENT_STOP = _STOPWORDS | {
    "abstract", "introduction", "section", "chapter", "figure", "table", "lemma",
    "theorem", "proof", "case", "vol", "pp", "doi", "http", "https", "www", "journal",
    "authors", "author", "year", "university", "press", "proceedings", "conference",
    "let", "thus", "hence", "therefore", "where", "given", "since", "every", "each",
    "image", "images", "terms", "harvard", "copyright", "license", "reserved",
    "page", "pages", "see", "also", "may", "must", "will", "can", "one", "two",
    "three", "first", "second", "third", "following", "above", "below", "shown",
    "results", "result", "method", "methods", "paper", "work", "approach",
}


def _content_terms(text: str, *, limit: int) -> list[str]:
    """Top content terms by frequency (longer, non-stopword, alpha-ish)."""
    from collections import Counter

    toks = re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", text.lower())
    counts = Counter(t for t in toks if t not in _CONTENT_STOP and len(t) > 3)
    return [t for t, _ in counts.most_common(limit)]


_EXTRACTIVE_STYLES = ("title_terms", "section_terms", "term_cloud")


def _extractive_query(chunk: dict, style: str) -> str | None:
    title = re.sub(r"[^a-zA-Z0-9 \-]", " ", chunk.get("title") or "")
    title_words = [w for w in title.split() if w.lower() not in _CONTENT_STOP and len(w) > 2]
    section = chunk.get("section_path") or ""
    # last meaningful section segment (drop the journal/container prefix)
    seg = section.split(">")[-1].strip() if ">" in section else section.strip()
    seg = re.sub(r"[^a-zA-Z0-9 \-]", " ", seg)
    seg_words = [w for w in seg.split() if w.lower() not in _CONTENT_STOP and len(w) > 2]
    terms = _content_terms(chunk.get("text", ""), limit=8)

    if style == "title_terms":
        # title head + a couple of distinct content terms not already in the title
        base = title_words[:6]
        extra = [t for t in terms if t not in {w.lower() for w in base}][:3]
        words = base + extra
    elif style == "section_terms":
        base = seg_words[:5] or title_words[:4]
        extra = [t for t in terms if t not in {w.lower() for w in base}][:3]
        words = base + extra
    else:  # term_cloud — content-term-only query (most divergent from the title)
        words = terms[:6]

    q = " ".join(words).strip()
    if len(q.split()) < 3 or len(q) < 10 or len(q) > 200:
        return None
    return q


def _gen_extractive(chunk: dict, style_name: str) -> dict | None:
    q = _extractive_query(chunk, style_name)
    if not q:
        return None
    return {
        "id": f"{NAMESPACE}{chunk['id']}::{style_name}",
        "query": q,
        "source_chunk_id": chunk["id"],
        "source_doc_id": chunk["doc_id"],
        "style": f"extractive:{style_name}",
        "title": chunk["title"],
    }


def _clean_query(raw: str) -> str:
    q = (raw or "").strip()
    # strip leading labels / quotes / markdown
    q = re.sub(r"^(query|question|problem)\s*[:\-]\s*", "", q, flags=re.I)
    q = q.strip().strip('"').strip("'").strip()
    # take first line only
    q = q.splitlines()[0].strip() if q else q
    return q


def _gen_one(client: httpx.Client, chunk: dict, style_name: str, style_instr: str) -> dict | None:
    prompt = _build_prompt(chunk, style_instr)
    try:
        resp = client.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                # NOTE: gemma3-QAT emits ~450-540 hidden "thinking" tokens before the
                # visible answer. ANY finite num_predict cap (even 512) truncates to an
                # EMPTY response (done_reason=length); leaving num_predict UNSET lets the
                # model reach its own EOS (done_reason=stop) and emit the query. Do NOT
                # add a `stop` sequence — a blank line inside the thinking ends it early
                # and yields empty output. Generation is ~1.5 q/s as a result.
                "options": {"temperature": 0.7, "top_p": 0.9},
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        q = _clean_query(resp.json().get("response", ""))
    except Exception as exc:  # noqa: BLE001
        log.debug("gen failed for %s: %s", chunk["id"], exc)
        return None
    if not q or len(q) < 8 or len(q.split()) < 2 or len(q) > 240:
        return None
    return {
        "id": f"{NAMESPACE}{chunk['id']}::{style_name}",
        "query": q,
        "source_chunk_id": chunk["id"],
        "source_doc_id": chunk["doc_id"],
        "style": style_name,
        "title": chunk["title"],
    }


# --- Cloud generation (Gemini-3.5-Flash; ~$ — T2 query-quality A/B) -----------
# Reuses gold_synth's gemini client (rag_common.gemini_embed) — same Vertex/ADC as
# the judge. HARD COST CAP enforced by --target in main(). The SAME dedup gate runs
# afterward, so contamination is impossible regardless of generator.
def _gen_one_cloud(chunk: dict, style_name: str, style_instr: str, model: str) -> dict | None:
    import time as _time

    from rag_common.gemini_embed import (_client, _is_fatal, _is_rate_limit,
                                          _parse_retry_after, llm_location)

    prompt = _build_prompt(chunk, style_instr)
    client = _client(location=llm_location())
    config = None
    try:
        from google.genai import types
        config = types.GenerateContentConfig(temperature=0.7)
    except Exception:  # noqa: BLE001
        config = None

    text = ""
    for attempt in range(4):
        try:
            if config is not None:
                resp = client.models.generate_content(model=model, contents=prompt, config=config)
            else:
                resp = client.models.generate_content(model=model, contents=prompt)
            text = (getattr(resp, "text", None) or "").strip()
            break
        except Exception as exc:  # noqa: BLE001
            import httpx as _httpx
            # Transient transport/network errors (server disconnect, conn reset,
            # timeout) are RETRYABLE — a single blip must NOT kill the whole run.
            # (A bare RemoteProtocolError once discarded a full 7.7k-query buffer.)
            _transient = isinstance(exc, _httpx.TransportError) or _is_rate_limit(exc)
            # Genuine fatal (ADC/auth/quota-exhausted) → propagate so main() STOPS and
            # reports rather than silently burning retries (per the T2 contract).
            if _is_fatal(exc) or not _transient:
                raise
            _time.sleep(min(_parse_retry_after(exc) or (2 ** attempt), 30))
    q = _clean_query(text)
    if not q or len(q) < 8 or len(q.split()) < 2 or len(q) > 240:
        return None
    return {
        "id": f"{NAMESPACE}{chunk['id']}::{style_name}",
        "query": q,
        "source_chunk_id": chunk["id"],
        "source_doc_id": chunk["doc_id"],
        "style": style_name,
        "title": chunk["title"],
        "approx_prompt_chars": len(prompt),
        "approx_out_chars": len(text),
    }


# --- Dedup gate ---------------------------------------------------------------
def dedup_against_eval(
    records: list[dict], eval_queries: list[str], embedder,
) -> tuple[list[dict], dict]:
    """Drop training queries too close to ANY eval query (lexical OR semantic).

    Returns (survivors, stats). Asserts 0 survivors above either threshold.
    """
    import numpy as np

    eval_tokens = [_tokens(q) for q in eval_queries]
    eval_vecs = _embed(embedder, eval_queries)  # (E, D), normalized

    stats = {"input": len(records), "jaccard_dropped": 0, "cosine_dropped": 0, "kept": 0}
    survivors: list[dict] = []

    # Batch-embed all candidate queries once.
    cand_vecs = _embed(embedder, [r["query"] for r in records])  # (N, D) normalized

    for rec, cvec in zip(records, cand_vecs):
        qt = _tokens(rec["query"])
        max_jac = max((_jaccard(qt, et) for et in eval_tokens), default=0.0)
        if max_jac > JACCARD_MAX:
            stats["jaccard_dropped"] += 1
            continue
        max_cos = float(np.max(eval_vecs @ cvec)) if len(eval_vecs) else 0.0
        if max_cos > COSINE_MAX:
            stats["cosine_dropped"] += 1
            continue
        rec["max_eval_jaccard"] = round(max_jac, 4)
        rec["max_eval_cosine"] = round(max_cos, 4)
        survivors.append(rec)

    stats["kept"] = len(survivors)

    # ASSERT (safety net at the canonical boundary, independent of filter gates):
    # no survivor exceeds either contamination threshold.
    if survivors:
        surv_tokens = [_tokens(r["query"]) for r in survivors]
        surv_vecs = _embed(embedder, [r["query"] for r in survivors])
        for r, st_, sv in zip(survivors, surv_tokens, surv_vecs):
            jac = max((_jaccard(st_, et) for et in eval_tokens), default=0.0)
            assert jac <= ASSERT_JACCARD_MAX, (
                f"contaminating survivor (Jaccard {jac:.3f}): {r['query']!r}"
            )
            if len(eval_vecs):
                cos = float(np.max(eval_vecs @ sv))
                assert cos <= ASSERT_COSINE_MAX, (
                    f"contaminating survivor (cosine {cos:.3f}): {r['query']!r}"
                )
    log.info("dedup: %s", json.dumps(stats))
    return survivors, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=12000,
                    help="approx number of POST-dedup queries to aim for")
    ap.add_argument("--method", choices=["extractive", "llm", "cloud"], default="extractive",
                    help="extractive (instant, $0, no GPU; DEFAULT), llm (local Ollama, "
                         "~0.5 q/s — slow), or cloud (Gemini via Vertex/ADC — COSTS $)")
    ap.add_argument("--gen-model", default="gemini-3.5-flash",
                    help="cloud generation model (only used with --method cloud)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--target-kept", type=int, default=0,
                    help="early-stop once this many KEPT queries collected (0=disabled)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-chunks", type=int, default=0,
                    help="cap chunks sampled (0 = derive from target)")
    args = ap.parse_args()

    TRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    eval_queries = load_eval_queries()
    log.info("loaded %d eval query strings (contamination set)", len(eval_queries))
    embedder = _build_embedder()

    chunks = load_chunks()
    random.shuffle(chunks)
    log.info("usable chunks: %d", len(chunks))

    # Oversample chunks ~1.4x target to absorb gen failures + dedup drops.
    n_chunks = args.max_chunks or min(len(chunks), int(args.target * 1.5))
    chunks = chunks[:n_chunks]

    records: list[dict] = []
    t0 = time.time()

    if args.method == "extractive":
        for i, ch in enumerate(chunks, 1):
            style = _EXTRACTIVE_STYLES[i % len(_EXTRACTIVE_STYLES)]
            rec = _gen_extractive(ch, style)
            if rec:
                records.append(rec)
            if i % 2000 == 0:
                log.info("extractive %d/%d (%d kept) %.0f q/s",
                         i, len(chunks), len(records), i / max(time.time() - t0, 1e-6))
    elif args.method == "cloud":
        # HARD COST CAP: only attempt --target jobs (no oversample) to bound spend.
        jobs = []
        for i, ch in enumerate(chunks[: args.target]):
            style_name, style_instr = PROMPT_STYLES[i % len(PROMPT_STYLES)]
            jobs.append((ch, style_name, style_instr))
        log.info("CLOUD gen: model=%s, %d jobs (HARD CAP=%d) — COSTS $",
                 args.gen_model, len(jobs), args.target)
        # Incremental insurance: flush each kept record to a partial file as it
        # arrives, so a hard crash/OOM can't discard the whole in-memory buffer.
        from rag_common.gemini_embed import GeminiFatalError, _is_fatal as _fatal_check
        partial_path = TRAIN_DATA_DIR / "queries.partial.jsonl"
        partial_f = partial_path.open("w")
        fail_streak = 0  # circuit breaker: a SYSTEMIC failure must abort, not skip 10k
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(_gen_one_cloud, ch, sn, si, args.gen_model)
                    for ch, sn, si in jobs]
            for n, fut in enumerate(as_completed(futs), 1):
                try:
                    rec = fut.result()
                except Exception as exc:  # noqa: BLE001
                    # Genuine fatal (ADC/auth/config) stops the run loudly; a lone
                    # straggler exception is logged and skipped (never fatal). But a
                    # RUN of consecutive failures is systemic (e.g. dead endpoint,
                    # missing env) → abort loudly rather than "succeed" with 0 kept.
                    if isinstance(exc, GeminiFatalError) or _fatal_check(exc):
                        partial_f.close()
                        raise
                    fail_streak += 1
                    if fail_streak >= 50:
                        partial_f.close()
                        raise RuntimeError(
                            f"{fail_streak} consecutive gen failures "
                            f"({type(exc).__name__}) — systemic, aborting") from exc
                    log.warning("gen future skipped (%s)", type(exc).__name__)
                    rec = None
                if rec:
                    fail_streak = 0
                    records.append(rec)
                    partial_f.write(json.dumps(rec) + "\n")
                    partial_f.flush()
                if n % 100 == 0:
                    rate = n / max(time.time() - t0, 1e-6)
                    log.info("cloud gen %d/%d (%d kept) %.2f q/s",
                             n, len(futs), len(records), rate)
                # Early-stop: once we have enough KEPT queries, cancel the
                # dead-weight tail (late-run quota collapse → ~0 yield over ~hours).
                if args.target_kept and len(records) >= args.target_kept:
                    log.info("reached --target-kept=%d at %d completed; cancelling %d stragglers",
                             args.target_kept, n, len(futs) - n)
                    for f in futs:
                        f.cancel()
                    break
        partial_f.close()
    else:
        jobs = []
        for i, ch in enumerate(chunks):
            style_name, style_instr = PROMPT_STYLES[i % len(PROMPT_STYLES)]
            jobs.append((ch, style_name, style_instr))
        with httpx.Client() as client:
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                futs = [ex.submit(_gen_one, client, ch, sn, si) for ch, sn, si in jobs]
                for n, fut in enumerate(as_completed(futs), 1):
                    rec = fut.result()
                    if rec:
                        records.append(rec)
                    if n % 250 == 0:
                        rate = n / max(time.time() - t0, 1e-6)
                        log.info("gen %d/%d (%d kept) %.2f q/s", n, len(futs), len(records), rate)
    elapsed = time.time() - t0
    log.info("generated %d raw queries in %.0fs (%.2f q/s)",
             len(records), elapsed, len(records) / max(elapsed, 1e-6))

    survivors, stats = dedup_against_eval(records, eval_queries, embedder)

    with QUERIES_PATH.open("w") as f:
        for r in survivors:
            f.write(json.dumps(r) + "\n")
    log.info("wrote %d queries -> %s", len(survivors), QUERIES_PATH)

    if args.method == "cloud":
        gen_backend = f"gemini:{args.gen_model}"
    elif args.method == "llm":
        gen_backend = f"ollama:{OLLAMA_MODEL}"
    else:
        gen_backend = "extractive"
    # Rough token accounting for cloud (chars/4 heuristic; in+out summed over records).
    approx_in_chars = sum(r.get("approx_prompt_chars", 0) for r in records)
    approx_out_chars = sum(r.get("approx_out_chars", 0) for r in records)
    summary = {
        "namespace": NAMESPACE,
        "method": args.method,
        "gen_backend": gen_backend,
        "gen_model": args.gen_model if args.method == "cloud" else None,
        "cloud_api_used": args.method == "cloud",
        "approx_input_tokens": approx_in_chars // 4 if args.method == "cloud" else 0,
        "approx_output_tokens": approx_out_chars // 4 if args.method == "cloud" else 0,
        "raw_generated": len(records),
        "dedup": stats,
        "kept": len(survivors),
        "elapsed_s": round(elapsed, 1),
        "eval_queries": len(eval_queries),
        "jaccard_max": JACCARD_MAX,
        "cosine_max": COSINE_MAX,
    }
    (TRAIN_DATA_DIR / "queries_summary.json").write_text(json.dumps(summary, indent=2))
    print("GEN_SUMMARY:", json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
