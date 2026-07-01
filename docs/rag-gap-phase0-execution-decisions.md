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
