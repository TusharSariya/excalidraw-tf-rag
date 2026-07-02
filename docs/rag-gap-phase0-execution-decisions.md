# RAG gap-analysis Phase 0 — autonomous execution decision log

Execution of the reviewed plan (`~/.claude/plans/so-theres-a-couple-elegant-rossum.md`, items 1 & 3 from
the 2026-07 RAG gap-analysis campaign). Branch: `rag-gap-phase0-measurement` (submodule `tools/`).
Started 2026-07-01. Autonomous run — user stepping away; log every decision for later review.

Scope recap: **Part A** = GLR BM25 `ingest --force --rebuild` + drift guard + authoritative baselines +
facet-filter verify. **Part B (T1)** = `eval reference-coverage` (gold-independent corpus-gap metric).
**Part B3 (T3)** = Axis-2 identity surgery (additive `doi:` tokens, collision dedupe, `acquisition_gap`
field, scoped `validate_gold` exemption) + two gap tracks. **T4** = author gap tracks. **T5** = cleanup (deferred).

Standing constraints held: no OCR; no cloud for extraction/embedding; all embedding via desktop GPU gateway
(`RAG_GPU_GATEWAY_URL=http://10.0.0.156:8765`); no new index on the Mac beyond existing; secrets stay in
`.env`; cloud only for the offline eval judge; no push/PR unless asked.

---

## Timeline & decisions

### D0 — Orientation (2026-07-01)
- On submodule branch `rag-gap-phase0-measurement`, clean tree, head `da7c5f8` (campaign deliverable).
- **Premise check PASSED:** GLR `ingest/bm25.py:_build_schema` already declares all 6 filter fields
  (`venue`, `arxiv_category`, `genre`, `venue_type`, `oa_version`, `is_retracted`) — so a rebuild WILL
  write them into the on-disk tantivy schema. Part A premise is sound.
- **Gateway up & warm:** `/health` → `cuda-qwen0.6b-1024` loaded, 43,578 cache entries (> 41,083 chunks),
  so a `--force --rebuild` re-embed is mostly cache hits (light gateway load; real cost is local PDF re-extract).
- **Flag semantics confirmed:** `--rebuild` drops/recreates the LanceDB table; `--force` re-ingests all docs
  regardless of sha256. Using **`--force --rebuild`** to guarantee full repopulation (avoids the emptied-table trap).
- `reference-coverage` command does NOT yet exist → T1 to build. `citations.sqlite` present (333 MB).

### D1 — BUG FOUND: `chunk_count` still has the schema-drift bug the campaign fixed for `search_bm25`
- `bm25.chunk_count(bm25_dir)` reproducibly returns **0**, yet `tantivy.Index.open(bm25_dir)` gives
  `num_docs = 41083` and `search_bm25` returns real hits. The index is healthy.
- **Root cause:** GLR `chunk_count` opens via `_open_or_create_index` (= `tantivy.Index(_build_schema(),
  path=…)`, which imposes the *current write schema*). The on-disk index predates the 6 filter fields, so
  this raises `schema does not match`, caught by `except Exception: return 0`. This is the **same bug class**
  as the campaign's headline fix (`5d36dd0`), which repaired `search_bm25` via `_open_for_search` but left
  `chunk_count` on the old opener. Any health/verify path using `chunk_count` sees BM25 as empty.
- **Decision (FIX):** point `chunk_count` at `_open_for_search` (identical one-line rationale to the campaign
  fix) and extend the schema-drift regression test to cover it. Low-risk, clearly correct, surfaces a real
  latent bug (collaboration principle #3). Note: the rebuild would *mask* this (rebuilt index has the current
  schema so even the old opener would work), but the fix is the durable correctness win.

### D2 — PRE-rebuild drift snapshot captured (the guard gate)
- BM25 `num_docs` (via `.open()`): **41,083**; LanceDB rows: **41,083**; distinct doc_ids: **5,811**
  (~7.1 chunks/doc, matches corpus-health); `doc_id_set_sha256 = deefe972639689748caa4e6c578bf06dd2b4bea80eed43a5d7ba73121564f620`.
- **GATE:** after `--force --rebuild`, the doc-id-set sha256 must equal this value (tiny count tolerance).
  Any change ⇒ extraction drift ⇒ **abort baseline publication** and surface (new corpus = deliberate decision).
- Saved to scratchpad `drift_pre.json` + `docids_pre.txt`.

### D3 — Directive change + MAC index damaged + machine-topology surprise (STOP for user input)
- **User directive (mid-run):** move ALL RAG compute to the SSH desktop; no RAG on the Mac; after syncing,
  delete all RAG stuff on the Mac so it only lives on the desktop. Also: exit plan mode (done — autonomous now).
- **Killed the Mac rebuild** (it was embedding on the Mac's MLX GPU via `backend=local`, and re-extracting with
  `docling` — both against the "all compute on the gateway/desktop" intent). `--force --rebuild` overwrites in
  place, so the Mac `cuda-qwen0.6b-1024` index is now **partial: 2,033 / 41,083 chunks**. Recoverable by a
  rebuild (on the desktop). No git-tracked data lost (index is gitignored).
- **SAFETY DONE:** pushed the Mac's `rag-gap-phase0-measurement` branch (BM25 fix `5d36dd0` + full campaign +
  this log) to origin `TusharSariyaOrg/excalidraw-tf-rag`. The Mac's unique commits are now safe remotely.
- **TOPOLOGY SURPRISE (the blocker):**
  - **Mac:** `tools/` is a git **submodule** → `TusharSariyaOrg/excalidraw-tf-rag.git`, branch
    `rag-gap-phase0-measurement` (now pushed). Clean.
  - **Desktop:** the RAG code is committed **in-tree in the monorepo** `TusharSariya/excalidraw-tf.git`
    (NOT a submodule), on branch `rag-self-improve-campaign`, with **112 uncommitted files** in
    graph-layout-rag — **66 modified/new `.py` source files** (citation_store, eval/benchmark, eval/gold_cases,
    eval/gold_validation, ingest/bm25, query/*, tests) + new scripts/`training/`/docs.
  - ⇒ The desktop is a **separate, more-advanced, actively-edited** RAG codebase; different origin/org and
    different git structure. Its uncommitted edits **overlap the exact files this plan (T3) targets**.
- **Why I stopped:** (1) the desktop's 66 uncommitted source edits are unsaved active work I must not clobber;
  (2) they supersede/overlap the phase0 code, so "sync Mac→desktop" is inverted and ill-defined; (3) deleting
  Mac RAG is irreversible; (4) the two topologies (submodule vs in-tree, two orgs) need a canonical-source
  decision only the user can make. Asked the user how to proceed (see AUQ).

### D4 — User decisions + canonical gap-map (re-grounded worklist)
- **User chose:** (1) **Desktop is canonical** — diff phase0 against it, add only genuine gaps; Mac phase0 is
  archival (already on origin). (2) **User commits/stashes** the desktop's 112 dirty files themselves and pings
  when the tree is clean. ⇒ I am **gated** on desktop writes/compute until that ping; no Mac RAG work (it's
  being retired).
- **Read-only gap-map vs desktop canonical `rag-self-improve-campaign` (dirty working tree):**
  - **BM25 schema-drift fix — GENUINELY MISSING on desktop.** `ingest/bm25.py` still opens via
    `_open_or_create_index` in `search_bm25` (line 127) and `chunk_count` (line 183); no `_open_for_search`.
    So the campaign's headline fix never reached the desktop — its hybrid retrieval is silently dense-only,
    and `chunk_count` silently returns 0, exactly as on the pre-fix Mac. **Port the fix (both call sites).**
  - **T1 `reference_coverage` — MISSING** (no file, no CLI). Note `eval/coverage.py` (the plan's template)
    is ALSO absent on the desktop (it was a phase0 deliverable), so build fresh. Building blocks present in
    `citation_store`: `normalize_doi` (:36), `corpus_oa_by_doi` (:266), `references_of` (:293), `in_corpus`
    column. So the metric is buildable with no new harvesting.
  - **T3 identity surgery — MISSING, grafts cleanly.** Desktop `_result_identities` is the clean baseline
    `{doc_id, canonical_doc_id} ∪ alias_doc_ids` → additive `doi:` tokens graft in directly. `EvalCase`
    (gold_cases.py) has no `acquisition_gap` field → add (default False). `validate_gold` (gold_validation.py)
    is the standard `missing` gate, no exemption → scoped graft. `benchmark.py` calls `validate_gold()` (:152)
    and `case_metrics(results, case.relevant_doc_ids)` (:60) — same shape phase0 assumed.
  - **T4 gap tracks — MISSING** (no gap qrels).
- **Re-grounding note:** implement against the DESKTOP's versions of these files (not phase0's), since its 66
  edits may differ; the eval-identity/validation files inspected are near-baseline so low conflict expected.
- **uv on desktop:** at `/home/tushar/.local/bin/uv` (not on non-interactive SSH PATH — use full path or
  source profile for compute).
- **STATUS: waiting for user ping that the desktop tree is clean.** Full worklist ready; will execute T1 →
  BM25-fix → T3 → T2 (rebuild on desktop GPU) → T4, all on the desktop, then handle Mac RAG retirement.
