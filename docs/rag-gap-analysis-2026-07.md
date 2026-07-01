# RAG gap analysis & improvement campaign — 2026-07

**Scope:** `graph-layout-rag` (GLR, graph-layout papers) + `rag-literature-rag` (RLR, RAG-design papers).
**Method:** evidence-led execution of the plan in `~/.claude/plans/…`, every decision timestamped in
[`rag-gap-analysis-execution-ledger.md`](./rag-gap-analysis-execution-ledger.md).
**Constraints held:** no OCR, no cloud for extraction/embedding, all embedding via the desktop GPU gateway,
no new RAG index on the Mac; cloud allowed only for the offline eval judge.

> **One-line result:** the two headline *algorithmic* levers (Qwen3-4B embedder, citation-graph ranking)
> are both **statistically NULL** on the current gold set. The one real win was an **infrastructure bug fix** —
> BM25 was silently disabled by schema drift, so all local "hybrid" retrieval had degraded to dense-only.
> This strongly confirms the campaign thesis: **these RAGs are measurement/substrate-bound, not algorithm-bound.**

---

## TL;DR for reviewers

| Lever | Result | Ship? | Evidence |
|---|---|---|---|
| **BM25 schema-drift fix** | **+0.074 nDCG@10** (dense 0.496 → hybrid 0.570 on NL) | ✅ **SHIPPED** | Restores production hybrid; was silently dense-only |
| Qwen3-4B embedder (vs 0.6B) | +0.0101, 95% CI [−0.003, +0.023] → **NULL** | ❌ keep 0.6B | 3× VRAM, 14s switch penalty, no significant gain |
| Citation-graph ranking (post-repair) | +0.0089, 95% CI [−0.001, +0.019] → **NULL** | ❌ | Keying repaired 22%→89%, still near-threshold null |
| Citation-keying repair (as infra) | keying 1,334→4,376; edges 100k→532k | ✅ done | Enables graph work; graph work itself didn't pay |
| Corpus full-text acquisition (Phase 2A) | recall ceiling already **0.955** | ⏸ deferred | Corpus covers ~95% of gold; won't move the benchmark |

**New eval instrumentation shipped (Phase 0):** `eval coverage` (recall-ceiling decomposition), corpus-health
full-text/metadata stratification, and a dual-backend (opencode-go + DeepSeek) credit-failover judge.

---

## 1. The measurement substrate (Phase 0) — build the ruler before optimizing

Three diagnostics were added because the campaign had previously nearly shipped a 0.90 nDCG that was really 0.67
(pooling bias). You cannot detect a win you can't measure.

- **`eval coverage`** — for every gold-relevant doc, classify it as `full_text` / `metadata_only` /
  `not_indexed` and report the **recall ceiling** (max achievable recall given corpus coverage).
  - Result: **catalog 0.955, pdf-deep-read 0.938.** The corpus already covers ~95% of gold, so the headroom
    is *ranking*, not *acquisition* — this demoted Phase 2A (full-text harvest) for near-term benchmark movement.
  - Caveat: gold was curated partly from the corpus, so this is an upper bound; true corpus-completeness needs
    reference-coverage (gold-independent), deferred.
- **corpus-health full-text stratification** — 57% of *indexed docs* are metadata-only, but only **8% of
  chunks** (stubs ≈1 chunk vs 7.1 mean). The abstract-only tail is thin — a depth limit, not a coverage gap.
- **dual-backend judge** — `GRAPH_RAG_JUDGE_BACKEND=auto` grades via opencode-go Zen and fails over to the
  DeepSeek API on credit/auth signals (401/402/403). Live-validated on opencode-go; DeepSeek arm unit-tested
  (needs `DEEPSEEK_API_KEY` to exercise live). Offline-only; never in the production path.

## 2. The BM25 schema-drift bug — the real win

**Symptom:** dense, hybrid, and hybrid_sparse2 returned byte-identical eval numbers.
**Root cause:** `ingest/bm25.py:search_bm25` opened the lexical index with `tantivy.Index(_build_schema(),
path=…)`, which validates the *current write schema* against the on-disk index. Indexes built before schema
fields were added (`venue`, `genre`, `is_retracted`, …) raised `schema does not match`, which `search_bm25`
**swallowed** → returned `[]` → every hybrid/sparse query (**eval AND the production `query` command**)
silently degraded to **dense-only**.
**Fix:** `_open_for_search` using `tantivy.Index.open()` (persisted schema); the core query fields exist in
every version. Regression test simulates schema drift. Applied to **both** GLR and RLR (RLR's was latent —
its index happened to be in sync).
**Impact:** with BM25 live, hybrid (0.570) beats dense (0.496) by **+0.074** on NL — the corpus is
BM25-dominant (BM25-only 0.537 > dense 0.496), which is exactly why prior campaigns promoted `sparse_weight=2.0`.
**Every local "hybrid" number measured while this bug was live is dense-only and must be re-measured.**

## 3. Qwen3-4B embedder — NULL

Measured (not assumed) on the gateway: 4B warm = 0.26 s/query, **4.6 GB VRAM**, dims 1024. VRAM fit is a
non-issue (the plan's FP8/Matryoshka work was unnecessary); the real cost is a **14 s cold-start / model-switch
penalty** and 2× embed time.

| profile | dense nDCG@10 | hybrid nDCG@10 |
|---|---|---|
| 0.6B | 0.496 | 0.570 |
| 4B | 0.476 | 0.581 |

Paired bootstrap CI on production hybrid: **+0.0101, 95% CI [−0.0028, +0.0228] → NULL.** On dense-only 4B is
*worse* (−0.020). **Verdict: keep 0.6B.** The plan's "+0.087 known win" did not replicate on the current gold
set (the 4B index is a faithful reembed, so the contradiction is real — likely a dense-only measurement against
an earlier/smaller synth gold set).

## 4. Citation-graph layer (the graphify / Leiden interest) — repair done, ranking NULL

**3A keying repair (executed):** resumed `cite enrich` lifted corpus keying **1,334 → 4,376** (~2.5% → ~89% of
the DOI-bearing corpus); citation edges **100k → 532k**, in-corpus edges **9,134 → 22,706**. This was worth
doing for graph completeness. (The S2 incoming pass hung near the end on rate-limiting; stopped cleanly, DB
integrity verified — the last ~800 papers are diminishing returns.)

**Payoff test:** `hybrid_citation` (PPR + bibliographic-coupling + co-citation, RRF-fused) vs the champion,
now on the repaired graph: **+0.0089, 95% CI [−0.0012, +0.0192] → NULL** (near-threshold).

**Boost-strength (α) sweep** — settles whether a better operating point exists:

| α (citation boost) | nDCG@10 | MRR |
|---|---|---|
| 0 (champion `hybrid_sparse2`) | 0.570 | 0.854 |
| 0.25 | 0.578 | 0.844 |
| 0.5 | **0.579** | 0.845 |
| 1.0 | 0.577 | 0.831 |
| 2.0 | 0.570 | 0.788 |

The peak is a **shallow +0.009 at α=0.5** (NULL), and stronger boost *degrades* back to the champion while MRR
falls monotonically — i.e. more citation weight actively hurts top-rank precision. **There is no operating point
where citation ranking clears the promotion bar.**

**Why this likely settles the Leiden question:** `hybrid_citation` already fuses the union of what a Leiden
community-boost (3B), a single-hop citation walk (3C), and bibliographic coupling (3D) would each contribute.
Those are strictly *softer* signals; if the full citation ranking is null at 89% keying, a Leiden soft-boost is
very unlikely to beat it — and the gold set is ~95% single-hop, removing the multi-hop case graph methods win on.

## 5. What to do next (recommendation)

1. **Rebuild the BM25 indexes with the current schema** (so faceted filter fields exist), then re-measure the
   hybrid baseline — the ledger's pre-fix hybrid numbers are dense-only and should be regenerated.
2. **Do NOT pursue** the 4B/8B embedder swap or a Leiden/citation-ranking layer for benchmark gain — both are
   null at current corpus/gold. (Keep the repaired citation graph; it's useful for the *research-tool* neighbor
   navigation, just not for ranking.)
3. **If more retrieval quality is genuinely needed**, the evidence says invest in the *substrate*: expand the
   gold set to expose currently-missing topics (so acquisition becomes measurable), or measure reference-coverage
   to find real corpus gaps — not another ranking algorithm.

## Settled NULLs (do not re-run without a new mechanism)

Cross-encoder/LLM/listwise rerankers · SPLADE (×5, dilutes −0.129) · RAPTOR/multi-hop (gold is single-hop) ·
contextual/TLDR prefixes · HyDE/multi_query on NL · citation-count prior · **Qwen3-4B/8B embedder swap (this
campaign)** · **citation-graph ranking at repaired keying (this campaign)** · SPECTER2 as the embedder (negative).
