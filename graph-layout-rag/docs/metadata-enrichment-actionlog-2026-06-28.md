# Metadata Enrichment — Action Log (2026-06-28)

Autonomous execution of the metadata-enrichment plan (`~/.claude/plans/humming-discovering-island.md`) on branch `graph-rag-metadata-enrichment` (off `rcll-coord-repack`). All times **NDT (Mac local)**; embedded log timestamps share the same wall clock. Back-populated from git commit times, run-log timestamps, and file mtimes.

## Timeline

| Time | Action | Type | Outcome / evidence |
|------|--------|------|--------------------|
| ~09:30 | Plan approved (ExitPlanMode); created branch `graph-rag-metadata-enrichment`; set up tasks T1–T9; probed desktop GPU box (RTX 3060 Ti **reachable**); confirmed local pytest (405 tests) | setup | — |
| ~09:31 | Recon subagent ground-truthed all plan anchor points (signatures, line numbers, reused utils) | investigate | all confirmed; found `ManifestItem.status` collision → renamed new field `oa_version` |
| ~09:32 | **T4** — `gpu_sync_to_remote.sh`: dedicated WAL-checkpointed `citations.sqlite` transfer (was excluded by `data/*.sqlite`) | code | `bash -n` clean |
| ~09:32 | **T2** subagent — `papers_meta`+`doc_specter2` tables, accessors, manifest filter fields + merge/canonical threading | code | 19 tests, suite 411 pass |
| ~09:33 | **T1** subagent — resumable `enrich_metadata.py` (OpenAlex+S2+arXiv) + `harvest enrich-metadata` CLI | code | 11 tests |
| ~09:33 | **T3** subagent — `format_results` join, `in_corpus_citation_stats`, `bibtex_for_doc`, `cite bibtex` | code | 21 tests, graceful-degrade verified |
| ~09:34 | Integration: full suite **435 pass** (2 pre-existing docling fails); CLI wiring OK; live dry-run validated real OpenAlex/S2/arXiv parsing (10 items, specter2+abstracts flowing) | verify | — |
| 09:34:27 | First full backfill started (background, `--workers 16`) | run | enrich-metadata.run.log |
| **09:35:02** | **commit c60e3f69f** — P1 metadata enrichment (store + surfacing) | **commit** | — |
| ~09:40 | **BUG CAUGHT during data validation:** OpenAlex coverage only **4% on DOI items** vs 92% on no-DOI; root-caused malformed `/works/https://doi.org/<doi>` → 429 → rate-budget starvation (proved via side-by-side curl) | diagnose | — |
| 09:43:40 | Killed buggy backfill (3226 rows, only 187 cbc) | run | — |
| 09:45 | Fixed to `/works/doi:<doi>`; verified 2 prior-failing DOIs now return cited_by/fwci/venue | fix+verify | — |
| **09:47:49** | **commit d8f151ebd** — OpenAlex DOI lookup fix | **commit** | — |
| 09:47:58 | Corrective `--force` re-run started (rps=8, S2 cooldown 20s); backed up manifest+sqlite first | run | enrich-metadata.force.log |
| ~09:36 | Watcher confirmed fix at scale: cbc **712 in 135s** (old ceiling 187) | verify | — |
| 10:12:48 | Backfill done: **82% enriched (4936/6050)**, 98% OpenAlex where DOI exists; 1179 deferred | run | cbc 4849, tldr 3441, specter2 4181 |
| 10:13–10:16 | Resume pass 1 → **0 progress** (all 1114 deferred) | run | enrich-resume1.log |
| ~10:16 | Diagnosed: OpenAlex `/works?search=` returns **`$0 budget, resets midnight UTC`**; `/works/doi:` still 200. No-DOI tail (715 JGAA + bibliography) blocked until reset | diagnose | — |
| ~10:18 | **T5** subagent — filter columns + LanceDB-WHERE prefilter + Tantivy fields + CLI flags + relaxed chunk-count guard | code | 18 tests, suite 453 pass |
| **10:22:07** | **commit fc76b12b9** — P2 filter columns (code) | **commit** | — |
| ~10:25 | Synced code + citations.sqlite to desktop; ran remote NL query (`query_remote.sh`) — fwci/venue/tldr/bibtex **surface**, but `cited_by_count` None (graph keying) | verify | — |
| ~10:30 | Added papers_meta fallback for `cited_by_count` + test | fix | suite green |
| **10:31:25** | **commit d0ba8fbea** — cited_by_count fallback | **commit** | — |
| ~10:35 | Re-sync + re-query: **cited_by 675 / 3 / 5 confirmed** end-to-end | verify | — |
| ~10:38 | **T7** subagent — default-OFF normalized citation/recency RRF prior (byte-identical at 0.0) | code | 6 tests, suite 460 pass; flagged 22% keying limit |
| **10:40:46** | **commit ec48df867** — P3 citation prior (default OFF) | **commit** | — |
| ~10:42 | Docs subagent — SKILL.md, README.md, research-doc outcomes | docs | — |
| **10:43:28** | **commit 1b0c45cf3** — docs | **commit** | — |
| ~10:45 | Wrote memory (`graph-rag-metadata-enrichment`); sitrep + AskUserQuestion (2 decisions) | report | — |
| 10:45–14:15 | **[user review / decision window]** | wait | user chose: rebuild-now-on-82% + fix-keying-then-eval |
| ~14:18 | Resumed: launched keying-investigation subagent + 4B-rebuild recon (in parallel) | investigate | embed cache 3.4G present; drop-before-build confirmed; disk 98% (reproducible→no backup needed) |
| ~14:24 | First 4B rebuild started (`ingest --rebuild`, docling default 4 GPU workers) | run | rebuild-4b.log |
| 14:26 | **CUDA OOM** — 4 docling vision-model workers (~1.5 GiB each) + embedder blow the 7.66 GiB card; index left partial (~50 docs) | diagnose | REMOTE_EXIT=1 |
| ~14:27 | Keying investigation returned: root cause = `doc_to_oa` built only from `papers.doc_id` (~22%); DOI-bridge lifts to 34% (query-time, no rebuild) | investigate | — |
| ~14:28 | Implemented Option A DOI-bridge + DB-backed test; measured **33% on live corpus** (2043/6071) | code+verify | suite 461 pass |
| **14:28:25** | **commit 5ace4c1ff** — DOI-bridge citation graph (22%→34%) | **commit** | — |
| ~14:30 | Recovery 4B rebuild started: **docling→CPU**, EXTRACT_WORKERS=2, expandable_segments | run | rebuild-4b-cpu.log |
| ~14:35 | Watcher: **healthy past OOM** (completed 1025 vs ~50) | verify | — |
| 14:42 | Box telemetry: 20 cores (load 5.79), GPU 0% util but 6759/7666 MiB used (embedder resident, 1080 free), RAM 2 GiB free; **96% of docs hit extraction cache** (144 docling runs / 3600 docs); rate ~300 docs/min | analyze | docling not the bottleneck |
| 14:43:52 | Handled per-PDF docling warning ("Page backend unloaded", `doi-10-1007-s00287`) — skipped via fallback, run continues | run | — |
| 14:47 | 4B recovery rebuild ongoing (~62%+), healthy | run | — |
| 15:26:50 | **WORKERS=2 run FAILED (cascade):** docling ProcessPoolExecutor worker died abruptly under memory pressure (swap 5 GiB) starting on `jgaa-3040-...`; pool never recovered → **1,422 docs cascaded to metadata-only fallback** (errors 0→1403). Index degraded: 24,574 chunks vs ~41k expected. Then wedged 37 min at 99.7% on the dead-pool tail | diagnose | rebuild-4b-cpu.log |
| 16:09 | Caught it via sit-rep check (errors=1403, frozen log); confirmed root cause (pool cascade, not poison-PDF); verified `extract_cache` (3,435 entries, success-path-only) will short-circuit good docs on re-run | diagnose | — |
| 16:11 | Killed wedged run; RAM 13→27 GiB free, swap 5 GiB→722 MiB, GPU emptied (confirms run was the hog); confirmed 4B embedder alone holds ~6.9/8 GiB → docling MUST stay on CPU | recover | — |
| 16:12:57 | **Clean relaunch: `--rebuild` EXTRACT_WORKERS=1** (CPU docling, 27 GiB RAM headroom → no pool to cascade); cached front flying at 100% embed-cache hits | run | rebuild-4b-w1.log (pid 207521) |
| 16:19:14 | **4B rebuild DONE in 6m16s** — 5,819 docs, **44,358 chunks** (>old 41,083 baseline; abstract backfills added chunks), errors/fallbacks **249** (genuine bad-PDF floor, NO cascade) | run | extract_cache short-circuited good docs; only ~1,440 prev-failed re-extracted single-threaded |
| ~16:20 | **4B index verified:** all 6 filter cols PRESENT; venue propagates to ALL chunks/doc (0 partial; multichunk 9/9); LanceDB `WHERE venue=` prefilter pushdown returns 381 chunks/16 docs; top venues sane (LNCS/TVCG/JGAA/ACM CSur) | verify | — |
| 16:21:59 | **0.6B rebuild launched** EXTRACT_WORKERS=1 — extraction 100% cache-hit (shared extract_cache keyed by chunk_profile, not embed profile → no docling re-run, zero OOM risk); 0.6B embed re-runs (backfilled chunks miss) | run | rebuild-06b-w1.log (pid 224653) |
| 16:31:30 | **0.6B rebuild DONE in 9m39s** — 5,819 docs, **44,358 chunks**, **249 errors (EXACT parity with 4B floor)** | run | both profiles identical: 44,358 chunks |
| ~16:45 | Monitor self-match bug: `pgrep -f cuda-qwen0.6b` matched the monitor's own cmdline → false "RUNNING" after real exit; killed stuck monitor | fix | (ingest had exited cleanly at 16:31) |
| ~16:46 | **0.6B verified:** 44,358 rows, all 6 filter cols PRESENT, JGAA prefilter 175 chunks/96 docs | verify | parity ✓ |
| ~16:47 | **Pinned `GRAPH_RAG_NL_EXPECTED_CHUNKS=44358`** in `.env` (re-enables exact guard) | config | — |
| ~16:48 | **E2E acceptance PASS:** `query_remote.sh` (real NL/4B thin-client) — guard passed at 44,358; results relevant; TLDR + cited-by(N in corpus) + PDF URLs surface | verify | **T5/P2 COMPLETE** |
| ~16:55 | User provided OpenAlex API key (gmail) + directive: keep both keys (hotmail+gmail), **auto-rotate on depletion** | input | — |
| ~16:58 | Probed OpenAlex: both keys have live budget; the "$0 resets-midnight-UTC" was the **unauthenticated mailto pool**, not the keys → tail never actually clock-blocked | diagnose | — |
| 17:0x | **Built+committed key-pool auto-rotation** (`OPENALEX_API_KEYS`, rotate on budget-429/spend-cap, retire+reset-budget, all-depleted give-up); 3 tests | **commit c89fac67a** | — |
| 17:02 | Smoke test stalled — root-caused to **S2 300s default cooldown** (my omission; orig run used 20s), not a rotation bug | diagnose | — |
| ~17:10 | **BUG 1:** `_oa_request` called bare `OPENALEX.request()` → every enrich lookup hit the **unauthenticated pool**, keys never used by enrichment | diagnose | — |
| ~17:15 | **BUG 2:** title-search took `results[0]` blind → wrong-paper matches ("Crossing Minimization"→LeNet, "Low power tech mapping"→LOFAR) would inject wrong venue/fwci into filter cols | diagnose | Jaccard 1.0 vs 0.0-0.06 cleanly separates |
| ~17:18 | Fixed both: route via `request_openalex` (keyed+rotation) + Jaccard title-match guard (top-5, ≥0.8, env-tunable); 4 tests; suite 466 pass (2 pre-existing docling fails) | **commit a68fe50fa** | — |
| 17:21 | **Tail resume DONE:** scanned 1114 → **enriched 748, terminal_miss 366, transient 0**; abstracts +533; **rotation FIRED in prod** (…laZK hotmail exhausted → rotated to gmail, no stall) | run | papers_meta 4936→6050; venue/cbc 4849→5597 (~92%) |
| ~17:23 | Synced manifest.json (venue 4,816/6,071) + citations.sqlite to desktop (targeted, not full sync — avoid index clobber) | ops | — |
| 17:24 | **Incremental rebuild launched** 4B (WORKERS=1, cached → ~1,160 docs/min, ETA ~3m) to push new venues into filter cols | run | rebuild-4b-w1b.log (setsid; nohup died w/ ssh) |
| 17:31 | 4B rebuilt — but venue coverage only **2,750/5,819** docs (≈old 2,696, +54 not +748) | verify-FAIL | — |
| ~17:35 | **BUG 3 (extract_cache staleness):** `extract_cache` stores `TextChunk` incl. the 6 filter fields, but its key is only (sha256,backend,chunk_profile) → a cache hit served venue baked in at the 16:xx pre-enrichment build; **1,875 freshly-enriched venues stayed NULL** in the index | diagnose | manifest had 4,816 ingested venues; index had 2,750 |
| ~17:37 | Fixed: `_apply_filter_fields()` overlays the live manifest item's filter fields onto cached chunks on every cache hit; regression test | **commit 4b51dec62** | suite green |
| 17:38 | Synced fixed run.py; relaunched 4B `--rebuild` with fix (1,500 docs/min) | run | rebuild-4b-w1c.log |
| 17:42 | **4B rebuilt with fix — venue coverage 2,750 → 4,637 docs** (+1,887, the dropped set recovered; gap vs 4,816 manifest = canonical dedup) | verify-PASS | 44,358 chunks |
| 17:44 | **0.6B rebuild with fix launched** (ETA ~4m) | run | rebuild-06b-w1c.log |
| 17:51 | **0.6B rebuilt with fix — venue 4,637 docs (EXACT parity with 4B)** | verify-PASS | 44,358 chunks |
| ~17:53 | **Final E2E:** prefilter on enriched venues — JGAA **588 docs**, IEEE TVCG 233, OPUS Passau 4; is_retracted all 0 (no-op OK); arxiv_category 693 docs | verify-PASS | **tail enrichment + filter cols DONE** |

## Commits (10)
```
c60e3f69f 09:35  P1 metadata enrichment — papers_meta store + query-time surfacing
d8f151ebd 09:47  fix OpenAlex DOI lookup form (4% → working)
fc76b12b9 10:22  P2 metadata filter columns (code)
d0ba8fbea 10:31  fix surface cited_by_count from papers_meta fallback
ec48df867 10:40  P3 citation/recency RRF prior (default OFF, eval-gated)
1b0c45cf3 10:43  docs — SKILL, README, research outcomes
5ace4c1ff 14:28  DOI-bridge citation graph doc_id->oa (22%->34%)
c89fac67a 17:0x  OpenAlex multi-key pool + auto-rotation on budget exhaustion
a68fe50fa 17:18  enrich via keyed pool + Jaccard title-search match guard
4b51dec62 17:37  re-apply filter fields on extract_cache hit (stale venue bug)
```
Both desktop profiles rebuilt to parity: 44,358 chunks, 4,637 venue docs each.

## Session 2 summary (OpenAlex key + tail)
Key request → 3 latent bugs found via shit-testing: (1) enrich path called bare
`OPENALEX.request()` so it never used any api_key (hit the metered unauth pool);
(2) title-search took results[0] blind → wrong-paper metadata (LeNet/LOFAR); (3)
extract_cache served stale filter fields from pre-enrichment builds. All fixed +
tested. Tail: 4,936→6,050 papers_meta (748 enriched, key rotation fired in prod
hotmail→gmail); index venue coverage 2,696→4,637 docs. P2 fully reconciled.

## Data runs
- `enrich-metadata.run.log` 09:34:27–09:43:40 — buggy backfill (killed)
- `enrich-metadata.force.log` 09:47:58–10:12:48 — corrective backfill, 82% enriched
- `enrich-resume1.log` 10:13:23–10:16:12 — resume (0 progress; budget exhausted)
- `rebuild-4b.log` ~14:24–14:26 — OOM (docling on GPU)
- `rebuild-4b-cpu.log` ~14:30– — recovery (docling on CPU), in progress

## Bugs found & fixed (not in original plan)
1. **OpenAlex DOI-URL 429** — `/works/https://doi.org/` → `/works/doi:` (4% → 98% DOI coverage).
2. **cited_by_count None on remote** — added papers_meta fallback (graph not built on desktop).
3. **CitationGraph keying** — DOI-bridge `doc_to_oa` (22% → 34%).
4. **Rebuild CUDA OOM** — docling vision workers on GPU; moved docling to CPU.

## T8 — SPECTER2 query-side A/B (CONFIRMED NEGATIVE, ships nothing)
SPECTER2 is a 768-d paper-level citation-trained space — dimensionally/semantically
incompatible with the 1024-d chunk dense index, so it can't re-retrieve. Implemented
instead as a **re-ranker** (`hybrid_specter_prf`, eval-only arm): a pseudo-relevance-feedback
centroid from the top-5 text-ranked docs' SPECTER2 vectors, every candidate doc scored by
cosine-to-centroid, multiplicatively blended into the text fusion score (same text-primary
blend as the citation arm; α tunable via `GRAPH_RAG_SPECTER_ALPHA`, α=0 = exact no-op).

A/B on catalog (n=49, `cuda-qwen0.6b-1024`, bootstrap-CI gate):
| arm | nDCG@10 | hit@5 | Δ vs hybrid_sparse2 (95% CI) |
|---|---|---|---|
| hybrid_sparse2 (prod baseline) | 0.6104 | 0.898 | — |
| hybrid_specter_prf α=0.2 | 0.5692 | 0.918 | **−0.0412 [−0.068,−0.017] SIG** |
| hybrid_specter_prf α=0.5 | 0.5482 | 0.918 | **−0.0623 [−0.099,−0.027] SIG** |

**Monotonically NEGATIVE** in α (0→0.0, 0.2→−0.041, 0.5→−0.062), significant at both. hit@5
*rises* (SPECTER pulls relevant docs into the pool) but the top-rank ordering degrades —
same failure mode as the citation arm: SPECTER optimizes topical/citation *relatedness*, not
query *relevance*, so it floats related-but-less-relevant papers above the exact answer. Runs:
`p3-specter-ab` (α=0.5), `p3-specter-ab-a02` (α=0.2). Disposition: kept as an eval-only arm
(not in `OFFLINE_STRATEGIES`, never in the production query path), like the other documented
negatives. **Nothing ships.**

## Known limitations / pending (externally gated)
- No-DOI tail (~1114, mostly JGAA) — blocked on OpenAlex search budget reset (**midnight UTC**); converges via a resume pass.
- Citation-prior coverage ceiling ~34% (rest are DOI-less, never in citation crawl) → P3 eval likely NULL; prior ships OFF.
- 0.6B rebuild + P3 eval gates (gold-set + LLM judge) — pending.
- 2 pre-existing docling test failures — unrelated to this work.
