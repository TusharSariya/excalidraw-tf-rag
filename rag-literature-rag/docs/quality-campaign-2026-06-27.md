# RAG Literature Quality Campaign — 2026-06-27

Overnight campaign: corpus expansion via new legal OA sources, a Qwen3-4B embedding
build with all heavy compute on the desktop GPU (query-from-Mac-hits-desktop), a judge
migration off dead Gemini, and a real late-chunking implementation. **This doc is the
running record; the final 4B-vs-champion verdict is pending the hole-filled
re-benchmark (see §5).**

## Status

- **Phase 1 — new sources + scrape: complete.** Added CORE/S2ORC/PMC(EuropePMC)/DOAB
  harvest modules (`harvest/{core_source,s2orc,pmc,doab}.py`), wired into `harvest/run.py`
  with `--max-*`/`--skip-*` flags (default OFF). 27 new tests green. Two scrape passes added
  **+377 net-new `ok` PDFs** (986 → 1363). Short of the 500 target — candidate-bound by
  relevance-gating + dedup against the existing corpus, not by caps. CORE is wired but needs
  `CORE_API_KEY` (keyless = `retryable_failure`); it was skipped for the scrape.
- **Phase 2 — Qwen3-4B on the desktop + remote query: build complete, eval in progress.**
  `cuda-qwen4b-1024` reembedded from `gemini-2-structure-v1` (21211 chunks) on the desktop
  CUDA box. `scripts/query_remote.sh` ports the graph-layout-rag remote-query pattern
  (SSH + parity preflight, fail-loud); `yarn rag-lit:query` now hits the desktop 4B,
  `rag-lit:query-local` keeps the local path.
- **Judge migration: complete.** Gemini account is dead (`invalid_grant: Account has been
  deleted`). Judge is now **DeepSeek V4 Flash via OpenCode Go** (`opencode-go/deepseek-v4-flash`),
  wired into `eval/judge.py` as an OpenAI-compatible HTTP backend.
- **Phase 3 — real late chunking: built + BENCHMARKED on TWO model classes.** On Qwen
  (`cuda-qwen4b-latechunk-v1`) it's **decisively worse** (−0.13–0.18 bpref, §4a) — the model is
  causal/last-token-pooled, the wrong class for the technique. On **jina-embeddings-v3** (the
  mean-pooled bidirectional model late chunking was designed for, §4b) it's a **wash** (±0.015
  bpref). Conclusion: the Qwen collapse was a model mismatch (confirmed), but late chunking
  **wins nothing even done right** → corpus is coverage-bound, not chunking-bound. Do NOT promote;
  no embedder switch (jina-plain < Qwen-4B-plain). (Earlier revision deferred this as
  "GPU-validated"; corrected — a smoke test ≠ a benchmark.)
- **Phase 4 — eval: complete. Verdict — stay on the 0.6B-contextual champion.** 4B-plain ties
  the champion (no robust win at higher cost); late chunking is worse than both. Judge migrated
  to DeepSeek V4 Flash via OpenCode Go.

## 1. The 8 GB "4B OOM" was a missing dependency, not a hardware limit

The repo lore said `cuda-qwen4b-*` was "skipped on RTX 3060 Ti 8 GB after OOM." Root cause
found: **`bitsandbytes` was not installed in the desktop venv**, so the 4-bit load silently
fell back to FP16 (~8 GB weights) and OOM'd. (0.6B got away with it — FP16 0.6B ≈ 1.2 GB.)
After `uv pip install bitsandbytes`, **Qwen3-4B 4-bit loads at ~2.9 GB used / ~4.9 GB free**
and embeds fine. Build via the **reembed path** (`ingest reembed --source-profile
gemini-2-structure-v1 --target-profile cuda-qwen4b-1024`) — no docling/re-extraction, and
apples-to-apples with the 0.6B-1024 baseline (same chunk text, same basis).

Throughput footgun: `RAG_CUDA_BATCH_SIZE=32` OOMs (activations); **batch 8 is the safe
ceiling** (~6.9 GB used, stable; ~3 texts/s, ~2 h for 21k chunks). "GPU not at 100%" at
small batch is CPU-side tokenization starving the GPU between forwards — but batch 8 is the
memory ceiling here, so the 4B reembed is effectively bandwidth/compute-bound on this card.

## 2. New sources (legal OA full-text)

Pattern mirrors `harvest/crossref.py`: `harvest_<src>(*, max_works, dry_run, workers,
existing_ids) -> list[ManifestItem]`, relevance-gated (`is_layout_relevant`), DOI-deduped,
downloaded via `download_to_file` (validates `%PDF`). Live-validated: **s2orc, europepmc,
doab work**; CORE needs a key. Per-source yield is candidate-bound (relevance + dedup):
europepmc was the top producer; doab near-zero for niche RAG topics. Shadow libraries
(Anna's Archive / LibGen / Sci-Hub) were declined — the tool stays legal-OA-only and these
four cover the coverage need. The +377 new docs are **not yet in any index** (folding them in
needs docling, not installed; deferred) — they don't affect the 4B-vs-0.6B A/B, which uses
the fixed 21211-chunk corpus.

## 3. Judge: Gemini → DeepSeek V4 Flash via OpenCode Go

Gemini (Vertex) auth is dead. DeepSeek V4 Flash (released 2026-04-24; 284B/13B MoE, 1M ctx,
$0.14/$0.28 per 1M tok) is reachable through OpenCode Go as an **OpenAI-compatible** endpoint.

- Endpoint `https://opencode.ai/zen/go/v1/chat/completions`, `Authorization: Bearer <key>`,
  **`User-Agent` header REQUIRED** (Cloudflare 403s the default urllib UA — this cost an hour).
  Key from `~/.local/share/opencode/auth.json` (`opencode-go.key`) or `OPENCODE_API_KEY`.
- **Do NOT use `opencode run`** for batch judging: it's a TTY/agent harness (PTY-sensitive
  like `agy`), ~6 s startup, and does NOT parallelize (concurrent calls contend → empty output).
  The direct HTTP path parallelizes cleanly: 8 workers, 0 errors, ~2/s+ sustained.
- Wired in `eval/judge.py`: `_judge_one_opencode` (retries 429/5xx/timeout/**empty**, never
  caches errors), `_judge_backend` (auto-routes `opencode-go/*`). Activate with
  `RAG_LIT_JUDGE_LLM_MODEL=opencode-go/deepseek-v4-flash`. Judge cache is model-keyed →
  DeepSeek grades don't collide with old Gemini grades. Calibration: curated-doc grade≥2
  rate = 1.0 on the smoke. 17 judge tests pass.
- The desktop needs `OPENCODE_API_KEY` passed via env (auth.json is Mac-only) for
  desktop-side judging (qrels-backfill).

## 4. Late chunking (real, additive, name-gated)

Implemented per the offset-provenance audit (offset *recovery* was only 14% viable → spans
tracked at construction time instead). Additive only, no regressions (+6 rag-common, +5
rag-lit tests; existing suites unchanged):
- `rag_common.local_embed`: `embed_local_texts_with_tokens` (ST `output_value="token_embeddings"`
  + tokenizer `return_offsets_mapping`) and `pool_span` (mean/max-pool tokens in a char span →
  Matryoshka-truncate → L2-normalize).
- `ingest/chunk.py`: defaulted window/span fields on `TextChunk` + `assign_late_chunk_windows`.
- `ingest/latechunk.py`: `is_latechunk_profile`, `embed_late_chunks`. Profile
  `cuda-qwen4b-latechunk-v1`.
- **GPU smoke PASSED (2026-06-27):** on real Qwen3-Embedding-4B, ST `token_embeddings`
  count **exactly matches** the tokenizer `offset_mapping` (19 == 19, ALIGNED), and
  `pool_span` yields a 1024-d L2-normalized vector (norm 1.0).

### 4a. Late chunking BENCHMARKED — NULL/NEGATIVE (2026-06-27, was previously deferred)

> **Correction:** an earlier revision of this doc let the GPU *smoke test* (token/offset
> alignment + `pool_span` correctness) stand in for a benchmark and deferred the build as
> "low expected value." That was wrong — a smoke test is not a retrieval measurement. The
> arm has now been built and benchmarked end-to-end. It loses decisively.

Wiring completed (the previously-deferred seam): `ingest/index.py:_embed_index_texts()` now
branches to `embed_late_chunks()` for latechunk profiles (bypassing the per-text embed cache,
whose chunk-text-only key is invalid when the vector depends on window context); `reembed.py`
got **doc-aligned batching** so a document's chunks never split across an embed batch (late
chunking needs the whole doc to build its window). +1 routing/cache-bypass test.

**Build:** `cuda-qwen4b-latechunk-v1` built on desktop CUDA by re-embedding the **exact
21,211-chunk set** of the 4B-plain arm (reembed from `gemini-2-structure-v1`) → guaranteed
corpus parity, no re-extraction. 2338 s @ 9.1 chunk/s, batch=1, ~3.5 GB VRAM, no OOM, 0 empty
vectors. Same chunk_id vs 4B-plain has cos **0.34–0.54** → late chunking moved the vectors a
lot (it did real work), but in a direction that **hurts retrieval**.

**Held-out test split, SAME frozen qrels as §5 (3-way, hole-robust):**

| arm (dense) | catalog bpref | catalog cond-nDCG | pdf bpref | pdf cond-nDCG | hole@10 | failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cuda-qwen4b-1024 (plain) | **0.426** | **0.693** | **0.489** | **0.731** | .27–.33 | 2 |
| cuda-qwen0.6b-contextual (champ) | 0.380–0.407 | 0.652–0.663 | 0.441–0.460 | 0.694 | .21–.37 | 2–3 |
| **cuda-qwen4b-latechunk-v1** | **0.296** | **0.523** | **0.308** | **0.501** | **.50–.53** | **10–11** |

**Verdict: late chunking is decisively WORSE — NOT a tie.** It trails plain 4B by **~0.13–0.18
bpref** and **~0.17–0.23 condensed-nDCG** on *both* tracks, with 5× the query failures. Raw
nDCG agrees (0.443/0.450 vs 0.612/0.658). The three guards are unanimous → **do NOT promote.**

**Why the result is robust to its own high hole rate:** latechunk has the highest hole@10
(.50–.53) — it surfaces many off-pool docs — so its *raw* nDCG is the most hole-distorted of
the three. But `bpref` ignores unjudged docs by construction and `condensed-nDCG` removes
them, and latechunk still ranks **last by a wide margin on both**. Even crediting every hole
as relevant cannot close a 0.13–0.18 bpref gap. Pooling+judging latechunk's holes would refine
the raw number but not the verdict.

**Why it fails (hypothesis → CONFIRMED in §4b):** Qwen3-Embedding pools by **last token only**
(`1_Pooling/config.json`: `pooling_mode_lasttoken=true`) and is **causal** (decoder LLM). Late
chunking instead **mean-pools** token spans of a window — a pooling method the model was never
trained for, over intermediate hidden states that aren't embeddings, on a causal model where
early chunks get no global context and late chunks get diluted. Late chunking was designed for
**mean-pooled bidirectional** encoders (Jina/BERT). §4b runs that control and confirms the
diagnosis.

> **Window-truncation caveat (found post-hoc):** `embed_local_texts_with_tokens` clamped every
> text to `MAX_EMBED_CHARS=3000` before embedding — so the Qwen latechunk windows were ~one
> chunk, i.e. this run mostly measured *pooling mismatch* with little real cross-chunk context.
> Fixed for §4b (window cap = model token budget). The Qwen verdict stands (the §4b control
> shows real windows don't rescue the technique), but the mechanism here was dominated by the
> last-token-vs-mean pooling mismatch, not long-window dilution.

### 4b. Late chunking on its HOME model (jina-embeddings-v3) — NEUTRAL, not catastrophic

Hypothesis test: run the same A/B on **jina-embeddings-v3** — mean-pooled, bidirectional,
8192-ctx, *the model class late chunking was invented for*. Both arms built by re-embedding the
**same 21,211-chunk set** (corpus parity), real full 20k-char windows (the §4a truncation bug
fixed), benchmarked on the **same frozen qrels**. This is the version-self-consistent comparison
(jina needs transformers <5; both jina arms run there).

Integration landed in `rag_common.local_embed`: jina family detection, `trust_remote_code`,
`retrieval.passage`/`retrieval.query` **task adapters** (keyed on doc/query mode), native 1024-d.
`embed_late_chunks` rewritten to embed+pool windows in **bounded batches** (a pathological
879-chunk doc stalled the all-windows-at-once version). Plus profiles `jina-v3-1024` /
`jina-v3-latechunk-v1`.

**Held-out test split, dense, SAME frozen qrels (the clean jina A/B):**

| metric | catalog plain | catalog **latechunk** | pdf plain | pdf **latechunk** |
| --- | ---: | ---: | ---: | ---: |
| bpref | 0.365 | **0.380** (+0.015) | **0.427** | 0.419 (−0.008) |
| condensed nDCG | 0.629 | **0.645** (+0.016) | **0.677** | 0.664 (−0.013) |
| raw nDCG | **0.564** | 0.540 | **0.617** | 0.565 |
| hole@10 | 0.37 | 0.44 | 0.34 | 0.42 |

**Verdict: on jina-v3, late chunking is a WASH** — catalog marginally favors latechunk (+0.015
bpref), pdf marginally favors plain (−0.008); all within ±0.02 = noise. Compare to §4a where on
Qwen it was **−0.13 to −0.18 bpref**. So:

1. **The model-mismatch diagnosis is confirmed.** Late chunking's catastrophic Qwen result was
   the causal/last-token pooling incompatibility — on a mean-pool bidirectional model it's
   neutral, exactly as predicted. (raw nDCG still dips for latechunk because it carries the
   higher hole@10; bpref/condensed, which are hole-robust, are flat.)
2. **But late chunking still doesn't WIN — even done right.** On *this* corpus, applied to its
   home model, it buys nothing. This reconfirms the campaign's central thesis: **the corpus is
   coverage-bound, not embedding/chunking-bound.**
3. **No reason to switch embedders:** jina-v3 *plain* (dense bpref 0.365/0.427) is **below**
   Qwen-4B-plain (0.424/0.488) and ≈ the champion — so jina is not an upgrade here regardless of
   chunking. Stay on the champion.

> Caveat: jina arms ran on transformers <5 (jina-v3 remote code is incompatible with tf 5.x);
> the jina-internal plain-vs-latechunk A/B is fully self-consistent. Qwen reference rows were
> re-run on the same machine state (doc vectors built on tf 5.11, queried on tf 4.x — negligible
> drift, identical weights). jina hybrid == dense (BM25 copied from gemini-2 didn't re-fuse), so
> dense is the clean comparison.

## 5. Eval — 4B vs 0.6B vs champion (PRELIMINARY)

Held-out test split (164 cases), existing Gemini-judged qrels (542 cases / 35k judged docs).

| arm | catalog dense | catalog hybrid | pdf dense | pdf hybrid |
| --- | --- | --- | --- | --- |
| cuda-qwen4b-1024 | 0.574 | 0.574¹ | 0.640 | 0.640¹ |
| cuda-qwen0.6b-1024 | 0.571 | 0.377 | 0.633 | 0.415 |
| cuda-qwen0.6b-contextual-v1 (champion) | 0.524 | **0.594** | 0.582 | **0.642** |

¹ 4B hybrid == its dense exactly → its BM25 index didn't transfer in the reembed; even
crediting 4B its dense score it does not beat the champion's hybrid.

**Clean read (dense): 4B ≈ 0.6B** (+0.003 catalog / +0.007 pdf — noise). No embedder-size win.

**The comparison is hole-limited**, so raw nDCG is not the metric to trust. Hole-robust
diagnostics (test split) decide it:

| metric | track | **4B-1024** | 0.6B-contextual (champion) |
| --- | --- | --- | --- |
| raw nDCG@10 (hole-distorted) | catalog | 0.568 | **0.587** |
| **bpref** (hole-robust) | catalog | **0.432** | 0.420 |
| **condensed nDCG** (hole-robust) | catalog | **0.674** | 0.663 |
| hole@10 | catalog | 0.44 | 0.32 |
| raw nDCG@10 | pdf | 0.634 | **0.637** |
| **bpref** | pdf | **0.500** | 0.478 |
| **condensed nDCG** | pdf | **0.734** | 0.707 |

The split is the tell: 4B surfaces **more unjudged docs** (higher hole@10), so raw nDCG
penalizes it (unjudged→0), while bpref/condensed (which ignore unjudged) favor it. On the
metrics this project's methodology endorses (bpref), **4B is marginally ahead on both tracks**
(+0.012 / +0.022). On hole-sensitive raw nDCG it's marginally behind. Either way the gap is
**~0.01–0.03 — a statistical tie.**

**Hole-filled re-judge attempted but throttled (not the decider).** `eval qrels-backfill
--model opencode-go/deepseek-v4-flash --in-place` ran but OpenCode Go **rate-limits to ~0.06/s
under sustained 6-worker load** (425 grades in ~2 h — the ~6k/hr / ~1.7/s ceiling, worsened by
429-backoff), so combo 1 never finished and the qrels weren't enriched in time. The 425 grades
are cached (`judge_cache.json`, model-keyed) for a future resumed run. The verdict does **not**
depend on it — bpref is hole-robust by construction.

### Verdict: do NOT promote 4B; stay on the 0.6B-contextual champion

Plain 4B-1024 **ties** the 0.6B-contextual champion (marginally ahead on bpref, behind on raw
nDCG, all within noise) at materially higher cost: ~1.5× query latency, ~2.9 GB VRAM model +
8 GB-class card required, slow embed (~3 texts/s, ~2 h/corpus), desktop-only (CUDA vectors not
Mac-portable), and BM25 didn't transfer in the reembed. This is the eval-findings thesis
confirmed once more: **embedder size is low-yield on this corpus.**

**The one untested config that could give a real (small) edge:** `cuda-qwen4b-contextual-v1` —
plain-4B already ties *contextual*-0.6B on bpref, so 4B + the winning contextual recipe might
edge ahead by ~0.01–0.03 bpref. Recommended **only if maximum quality regardless of operational
cost**; otherwise skip. (Build needs context-gen via the DeepSeek/opencode backend wired into
`rag_common.local_llm`, or a Mac-side cache warm — Gemini is dead.)

## 6. Operational footguns logged

- bitsandbytes missing → 4B FP16 → OOM (the whole "4B doesn't fit" myth).
- `RAG_CUDA_BATCH_SIZE=32` OOMs on 8 GB; use 8.
- OpenCode Go direct HTTP needs a `User-Agent` (else CF 403); `/zen/go/v1` not `/zen/v1`
  (the latter is a separate, unfunded product → CreditsError).
- `opencode run` is unsuitable for batch (no parallelism, TTY-sensitive) — use direct HTTP.
- `eval benchmark --split test` needs `data/eval/tune_test_split.json` present **and**
  `RAG_LIT_SYNTH_GOLD=1` (else silently falls to curated-42); `--run-dir` must be empty or
  pass `--resume`.
- Desktop source tree lags the Mac — sync edited modules (`eval/judge.py`) before running
  desktop-side jobs that use them.
- The contextual cache key is `{ctx_version}:{LLM_backend}:{LLM_model}:{sha}:{idx}` —
  embedder-agnostic, but format drift (old `rag-v2:{sha}:{idx}` entries) means most of the
  existing cache won't hit current code.
- **OpenCode Go judge throughput:** ~1.7/s ceiling (≈6k/hr); sustained 6-worker load drops to
  ~0.06/s with 429-backoff. Fine for small incremental hole-fills with patience; a full
  re-judge (tens of thousands of pairs) needs a higher-tier key or many hours. Judged grades
  are model-keyed-cached, so a throttled backfill resumes cheaply.
- Many parallel SSH calls to the desktop intermittently return exit 255 (transient); retry
  with a short delay. The box stayed up throughout.
- Desktop source tree must be synced per-file before desktop-side runs use new code
  (`eval/judge.py`, `rag_common/local_embed.py`, `ingest/{latechunk,chunk}.py`,
  `embed_profiles.toml` were each scp'd this campaign).
