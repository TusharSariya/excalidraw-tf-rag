# RAG gap-analysis — autonomous execution ledger

> Running log of an overnight autonomous execution of the plan in
> `~/.claude/plans/so-theres-a-couple-elegant-rossum.md`
> (deliverable roadmap → `docs/rag-gap-analysis-2026-07.md`).
> Every decision, experiment, block, and pivot is timestamped so it can be reviewed.
> All timestamps are **NDT (America/St_Johns)** unless noted.

Branch: `rag-gap-phase0-measurement` (stacked on `gpu-gateway-vram-scheduler-eval`, which carries the CI/ruff/hooks baseline, 12 commits ahead of `master`).

Constraints held throughout: **no OCR**, **no cloud for extraction/embedding**, **all embedding via the desktop gateway** (`http://10.0.0.156:8765`), **no RAG DB/index on the Mac beyond what already exists** (index migration is deferred), secrets stay in `.env`. Cloud is allowed **only** for the offline eval judge (never touches production retrieval).

---

## 2026-07-01 — Session start (~03:14 NDT)

### Environment ground truth (verified, not assumed)
- **Desktop SSH** (`ssh desktop`) → `tushar-ubuntu`, reachable.
- **GPU** (RTX 3060 Ti, 8192 MiB): idle at session start — 17 MiB used, 0% util.
- **Gateway** `http://10.0.0.156:8765/health` → `status:ok`, `sched_mode:affinity`, `loaded_models:[]`, `cache_entries:2158`, `ram_cache_4b:off`, `dispatcher_alive:false`.
- **Submodule** `tools/` (`TusharSariyaOrg/excalidraw-tf-rag`): local `master` at PR #2 merge; CI branch `gpu-gateway-vram-scheduler-eval` is +12 (unmerged). Work branched from the CI branch.
- Gateway daemon restart needs sudo → per user instruction, **use SSH to the desktop for gateway/GPU testing**; do not attempt daemon restarts.

### Decisions locked at start
| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Branch off the CI branch, not `master` | Inherit ruff config + pre-push hooks + integration-test marking so overnight commits stay CI-clean. |
| D2 | Cross-tool docs live in new `tools/docs/` | Ledger + deliverable are cross-tool (RLR + GLR); per-tool `docs/` don't fit. |
| D3 | Execution order: Phase 0 → Phase 1 validation → Phase 3A → (Phase 2 as time allows) | Phase 0 is foundational + self-contained + low-risk (diagnostic, can't break prod). Phase 1 is the measured known-win. Phase 3A (citation keying) unblocks the graph layer the user is most interested in. Phase 2 full-text harvest is network-bound/long — run opportunistically. |

---

## 2026-07-01 ~03:30 NDT — Phase 1 empirical validation (4B vs 0.6B through the gateway)

Ran a timed embed probe (`scratchpad/embed_probe.py`, Mac-side urllib, novel nonce'd text → cache miss) against the live gateway while polling `nvidia-smi` on the desktop. Measured, not assumed:

| Metric | `cuda-qwen4b-1024` | `cuda-qwen0.6b-1024` |
|---|---|---|
| Cold-start (model load) | **14.2 s** | 8.4 s (also evicted 4B) |
| Warm batch (8 texts) | 0.293 s (36.6 ms/text) | 0.202 s (25.2 ms/text) |
| Warm query (n=1) | **0.264 s** | 0.145 s |
| Output dims | 1024 ✓ | 1024 ✓ |
| Steady VRAM (loaded) | **4625 MB** | 1585 MB |

### Findings that reshape Phase 1
- **F1 — VRAM fit is a NON-issue.** 4B @ 4-bit steady = **4.6 GB**, well under 8 GB. The plan's FP8-quant + Matryoshka-768 (Phase 1.1) are **unnecessary for fit** and can be dropped. Confirms the shelving reason was latency, not memory.
- **F2 — The real blocker is model-switch thrash, not cold-start-once.** Loading 0.6B **evicted** 4B entirely (`loaded:['cuda-qwen0.6b-1024']`, `parked:[]`, because `ram_cache_4b:off`). So the plan's two-config routing (NL→4B, keyword→0.6B, Phase 1.2) would pay a **~8–14 s cold reload on every query-type switch** — a latency disaster under mixed traffic. This is THE Phase 1 engineering problem.
- **F3 — Warm 4B is plenty fast.** 0.264 s query / 37 ms per indexed text warm. If 4B is kept warm, latency is a non-issue.

### Phase 1 re-decision (D4)
The warm-serve fix is the whole game. Three candidate policies to evaluate (all need the eviction/affinity policy in `gpu-gateway/src/gpu_gateway/scheduler.py`):
1. **Single-model 4B, kept warm** — route *everything* to 4B, retire 0.6B from the query path. Simplest, no thrash, 4.6 GB. Cost: lose 0.6B's keyword edge (need to quantify magnitude — may be small).
2. **Co-resident 4B + 0.6B (+reranker)** — 4.6 + 1.6 + ~1.5 ≈ 7.7 GB; likely too tight once activation memory is added. Needs a hard VRAM test.
3. **0.6B co-resident default + 4B RAM-parked** (`ram_cache_4b:on`) — NL queries pay a RAM→VRAM swap (faster than the 14 s disk cold-load) instead of a full reload.
Deploying any of these needs a **daemon restart (sudo-gated)**. Per the storage/no-sudo constraint I'll **prototype + A/B the policy on the desktop in user space** (alt gateway instance on a spare port) and hand off the exact config + restart step, rather than restarting the prod daemon.
Storage lever noted (user): if desktop/Mac disk fills, delete Ollama models first (HyDE/contextual are deprioritized NULLs, so safe).

## 2026-07-01 ~03:55 NDT — Phase 0.2 SHIPPED: corpus-health full-text stratification

Built `audit_fulltext_stratification(profile)` in `graph-layout-rag/.../eval/corpus_health.py`, wired into `run_audit`, behind a patchable `_manifest_by_id()` seam. Classifies every indexed doc/chunk via the manifest `status`/`localPath` predicate (same as `gold_validation.py:21,33`) into full-text / metadata-only / unkeyed, reports doc- and chunk-weighted fractions, and warns (not critical → no CI exit-1) when metadata-only docs ≥40%. 5 unit tests (monkeypatched, hermetic) pass; ruff clean.

**Design choice (D5):** kept this audit *composition-only* (static, no retrieval). The plan's "nDCG per stratum" split needs a retrieval run, so it belongs in the `eval recall` command (Phase 0.1) — not in the fast static health check.

**Validated on the live index** (`cuda-qwen0.6b-1024`, 41,083 chunks / 5,811 indexed docs):
- **2,513 full-text / 3,298 metadata-only / 0 unkeyed docs → 57% metadata-only docs.** (Higher than the plan's 54% estimate; 5,811 indexed < 6,071 manifest ⇒ ~260 failed/unindexed.)
- **But only 8% of *chunks* are metadata-only** (stubs ≈1 chunk each vs 7.1 mean for full-text). 
- **Reframing (important):** 57% of *documents* are abstract-only, yet they're only 8% of *retrievable text*. The metadata-only set is a **thin discoverable long tail**, not a bulk-quality problem. This lowers the expected ROI of blanket full-text acquisition (Phase 2A) and argues for **targeting acquisition at metadata-only docs that actually appear in gold/recall gaps**, not all 3,298.
- **0 unkeyed** ⇒ manifest↔index keying is clean (distinct from the *citation-graph* keying gap at 2.5%, which is a different key space — OpenAlex/S2 IDs).

## 2026-07-01 ~04:20 NDT — Phase 0.1 SHIPPED: `eval coverage` (recall-ceiling decomposition) + a plan-reshaping result

Built `eval coverage` (`eval/coverage.py`, wired in `cli.py`, 2 hermetic unit tests, ruff clean). For every gold-relevant doc it reports the stratum (full_text / metadata_only / indexed_unkeyed / **not_indexed**) and computes the **recall ceiling** = fraction of relevant docs recoverable at all. No retrieval, no gateway — pure corpus/manifest read. Reuses `cases_for_track` + the `corpus_health` manifest seam.

**Live result (`cuda-qwen0.6b-1024`):**
| Track | cases | rel docs | full_text | metadata_only | not_indexed | **recall ceiling** |
|---|---|---|---|---|---|---|
| catalog | 49 | 176 | 68% | 27% | **8 (5%)** | **0.955** |
| pdf-deep-read | 47 | 128 | 94% | — | **8 (6%)** | **0.938** |

### This partially FALSIFIES the plan's Phase 2A ROI (important — the kind of thing "don't assume" is for)
- The corpus already covers **~94–96%** of every gold-relevant doc. Only **8 relevant docs per track are missing from the index.** So **full-text acquisition (Phase 2A) can lift measured recall by at most ~4–6%**, and most of that headroom is *acquiring 8 specific missing docs*, not the 3,298 metadata-only stubs (those are already indexed on their abstracts, i.e. `metadata_only`, not `not_indexed`).
- **Corollary:** the metadata-only 57% (Phase 0.2) is a *depth* limit (abstract vs full text for docs already present), not a *coverage* gap. Converting stubs→full-text helps ranking *precision* on those docs, but won't add new recall for the current gold set.
- **Where the headroom actually is:** the gap between *actual* recall@k and the 0.955 ceiling is entirely **ranking-improvable**. And the one measured non-NULL ranking lever is the **4B embedder (Phase 1)**. So the evidence now points even harder at Phase 1 as top priority, and demotes Phase 2A for near-term benchmark movement.

### Honest caveat (D6) — the 0.955 ceiling is an upper bound, likely inflated by gold-construction bias
Gold relevant-doc sets were curated partly *from* the corpus, so "relevant docs are in-corpus" is close to tautological; the 8 not-indexed docs are the human-knowledge leak-in. Therefore:
- The **true corpus-completeness** question ("what relevant literature are we missing entirely?") is **NOT** answered by gold recall ceiling. It needs **reference-coverage** (% of a paper's *cited* DOIs present in corpus), which is gold-independent and reads `citations.sqlite`. That's the honest version of Phase-0.1's "reference coverage" and is deferred to a `coverage --references` mode (needs the citation graph; overlaps Phase 3A keying).
- Net: `eval coverage` correctly says *acquisition won't move the current benchmark*; it does **not** say *the corpus is complete*. To make acquisition measurable we must either **expand the gold set** to include currently-missing topics or **measure reference-coverage**. Logged as a required follow-up before investing in Phase 2A.

## 2026-07-01 ~04:45 NDT — Phase 0.3 SHIPPED: dual-backend HTTP judge (opencode-go + DeepSeek) with credit-failover

Ported RLR's direct-HTTP opencode judge into GLR `eval/judge.py` and extended it to a **failover pair**, opt-in via `GRAPH_RAG_JUDGE_BACKEND` (`opencode` | `deepseek` | `auto`). `auto` grades via opencode-go Zen, and on a **credit/auth signal (HTTP 401/402/403)** raises `_JudgeCreditError` and fails over to the DeepSeek API — distinct from the transient set (`408/409/425/429/5xx`, which retries with jittered backoff). Persistent all-backend failure raises (so `judge_pool` records a retryable miss, never a fake grade-0 — the pooling-bias trap). Legacy `local_llm`/gemini path untouched when the env is unset. `judge_model()` now returns the backend-correct model id. Added the keys to `.env.example`.

- **Live-validated**: a real opencode-go call graded a Sugiyama paper vs "layered graph drawing crossing minimization" → **GRADE=3** with a coherent reason. End-to-end HTTP path works on the available `auth.json` opencode-go key.
- **DeepSeek failover**: unit-tested (mocked) — no `DEEPSEEK_API_KEY` on this box, so the live DeepSeek arm can't be exercised tonight; **provisioning that key is the one remaining step** before `auto` can actually fail over in anger. Model-name gotcha baked in: Zen's `deepseek-v4-flash` ≠ DeepSeek-API `deepseek-chat` (configurable via `GRAPH_RAG_DEEPSEEK_JUDGE_MODEL`).
- 7 new mocked unit tests pass; 25 existing judge tests still green (no regression); ruff clean.
- Cache honesty note (D7): in `auto` mode the cache key stays the single `GRAPH_RAG_JUDGE_HTTP_MODEL` string even if a grade actually came from the DeepSeek arm — keeps the qrels single-model, at a small fidelity cost documented in code.

### Reembed status (background): 4B index build in progress
`ingest reembed 0.6B→4B` running (copied BM25, embedding 41,083 chunks at batch 256 via gateway). ~18/161 batches at ~17 s/batch → ETA ~40 min. This unblocks the Phase 1 4B-vs-0.6B re-measurement on the **existing catalog-nl qrels** (no judge needed for that comparison).

## 2026-07-01 ~05:05 NDT — Phase 1 experiment staged + two navigation findings

**Confirmed the NL-track eval methodology** (for the 4B-vs-0.6B money experiment): the `catalog-nl` track = 49 curated + **126 synth** cases (175 total), so a correct run needs `GRAPH_RAG_INCLUDE_SYNTH=1` (loads the synth query strings) + `--qrels data/eval/qrels/catalog-nl/qrels.json` (overlays the de-biased judgments) + `--strategy dense` (where the prior +0.087 was measured). `benchmark --qrels PATH` exists. The desktop `gpu_embedding_bakeoff.sh` is NOT usable — it reembeds/evals on the GPU box's *stale pre-submodule checkout*; I run locally against the Mac index instead (embedding still routes through the gateway).

**Finding — graphify does not cover the RAG tools.** The `graphify-out/graph.json` (which the hook points at) was built over the **excalidraw TypeScript monorepo**, not `tools/*` Python. A query for citation/Leiden code returned only terraform-layout TS nodes. So graphify is not a navigation aid for this work; direct reads + subagents are. (Noted so the hook's grep-gating doesn't mislead future sessions.)

**Reembed** at 33/161 batches, 4B index 791 MB and growing; ETA ~30 min.

## 2026-07-01 ~05:20 NDT — Phase 3A GO: citation-keying repair started (subagent brief → execution)

A code+DB subagent brief settled the go/no-go: **the keying gap is an incomplete enrich run, not a broken method.** Key facts (live DB):
- `papers` total 53,136 (mostly external reference/citer nodes by design); **corpus-keyed = 1,334.** The oft-quoted "2.5%" uses the 53k denominator — the honest number is **~27% of the 4,907 DOI-bearing corpus items keyed**, ceiling **~81% of corpus** (4,907 distinct DOIs / 6,071 items). No arXiv column exists; arXiv would add only 7. 942 items are title-only (optional low-precision tail).
- Single write path `harvest/cite_enrich.py:enrich_citations` keys only DOI-present items that OpenAlex resolves; the run simply never finished → **3,580 unkeyed corpus DOIs**, of which **503 are keyable with ZERO API calls** (a `papers` row already exists for that DOI) and **3,077 need free OpenAlex singleton lookups** (~1 min at the configured RPS 100 / 16 workers; `OPENALEX_API_KEY` + rotation pool present).
- `paper_aliases` + `cite_provenance` are empty (predate the alias code). A query-time bridge `citation_rank.py:_bridge_doc_to_oa_by_doi` already lifts coverage ~22%→~34% in memory; **Phase 3A persists that + adds the new fetches.**
- **3B Leiden scaffolding:** `citation_rank.py` already has `CitationGraph`, `bibliographic_coupling`, `co_citation`, `personalized_pagerank` (pure-Python). **No community detection exists**; `igraph`/`leidenalg`/`networkx` are NOT deps — 3B must add one.

**Decision D8 — execute 3A via the tested `cite enrich` command, not a hand-rolled script** (collab principle: reuse tested code). Staged: `cite enrich --no-incoming` first (fast OpenAlex keying + outgoing edges) → verify keying jump → then the slower S2 `--incoming` pass to revive co-citation. DB backed up (`citations.sqlite.bak-2026-07-01`) first. Runs independently of the gateway/reembed.
**Next experiment enabled:** once keyed to ~81%, A/B the *existing* `hybrid_citation` strategy vs `hybrid` on the NL qrels — a clean test of whether citation ranking was coverage-limited (prior NULLs were at ~22–34% keying). Only build 3B Leiden if that shows signal or as an explicit soft-additive probe.

## 2026-07-01 ~10:30 NDT — 🔴 MAJOR BUG FOUND & FIXED: BM25 silently dead → all local hybrid eval was dense-only

While running the Phase 1 4B-vs-0.6B experiment, dense / hybrid / hybrid_sparse2 returned **byte-identical** numbers per profile. Root-caused (shit-testing, not assuming):
- **`ingest/bm25.py:search_bm25` was returning `[]` for every query.** `_open_or_create_index` opens with `tantivy.Index(_build_schema(), path=…)`, which validates the *current write schema* against the on-disk index. The BM25 indexes (built Jun 24) predate schema fields added later (`venue`, `genre`, `venue_type`, `oa_version`, `is_retracted`, …), so the open raised `ValueError: Schema error: … does not match`, which `search_bm25` **swallowed** → empty BM25.
- **Blast radius:** every hybrid/sparse query on this Mac — in **eval AND the production `query` command** — silently degraded to **dense-only** for any index built before the schema change. Direct `tantivy.Index.open()` proved the index itself is fine (41,083 docs, real hits). The original author documented choosing silent-disable ("Dense retrieval remains valid; the next rebuild creates the new schema") — but that means a schema bump silently kills hybrid until a full re-ingest.
- **Fix:** added `_open_for_search()` using `tantivy.Index.open()` (reads the *persisted* schema; the 5 core `_QUERY_FIELDS` exist in every version). `search_bm25` now uses it; the `except` is narrowed to genuinely-unreadable indexes. Regression test simulates schema drift and asserts search survives. 7 bm25 tests pass, ruff clean. **This is pre-existing, not my regression** (untouched query/ingest code) — surfaced + fixed because it blocked the Phase 1 measurement and corrupts every local hybrid result.
- **Immediate proof the fix matters:** with BM25 live, **BM25-only nDCG@10 = 0.537 on NL, beating dense's 0.496** — the corpus is BM25-dominant, exactly why prior campaigns promoted `sparse_weight=2.0`. Every local "hybrid" measurement taken while this bug was live was actually reporting dense-only and is **suspect** (re-measure before trusting).

### Phase 1 (4B) — dense-only comparison so far (both apples-to-apples dense; NL track, 175 cases)
| profile | dense nDCG@10 | MRR | R@5 | misses |
|---|---|---|---|---|
| 0.6B | **0.496** | 0.699 | 0.189 | 7 |
| 4B | 0.476 | **0.775** | 0.173 | 13 |
- **4B does NOT beat 0.6B on dense NL** (−0.020 nDCG@10) — contradicts the plan's "+0.087" premise (which I could not find persisted; likely a different/earlier gold set). 4B ranks its hits higher (MRR +0.076) but has 6 more complete misses, mostly hard/single-relevant synth cases. Miss sets are largely disjoint (4 shared). Full hybrid comparison (with fixed BM25) running now — that's the real ship-decision number.

## 2026-07-01 ~10:50 NDT — ⛔ Phase 1 VERDICT: 4B does NOT replicate as a win — keep 0.6B

With BM25 fixed, ran the correct 4B-vs-0.6B comparison on the **production strategy** (`hybrid_sparse2`, NL track, 175 cases):
| profile | hybrid nDCG@10 | MRR | R@5 | misses | dense nDCG@10 |
|---|---|---|---|---|---|
| 0.6B | 0.570 | 0.854 | **0.216** | 8 | **0.496** |
| 4B | **0.581** | 0.859 | 0.206 | 7 | 0.476 |

**Paired bootstrap CI (4B−0.6B, production hybrid): +0.0101, 95% CI [−0.0028, +0.0228] → NULL** (64 cases better / 49 worse / 62 tied). On dense-only 4B is actually −0.020.

**Decision D9 — DO NOT SHIP 4B; keep 0.6B as production.** Rationale:
- The gain is statistically **null** and the point estimate (+0.011) is below the plan's own +0.03 promotion bar.
- 4B costs **3× VRAM** (4.6 vs 1.6 GB), **~2× embed time**, and a **14 s cold-start / model-switch penalty** under the mixed routing the plan proposed.
- **The plan's headline "+0.087 known win" does NOT replicate on the current gold set.** Most likely it was a dense-only measurement against an earlier/smaller synth gold set (synth evolved 42→531). BM25 was likely also broken/absent in whatever produced it. My 4B index is a faithful reembed (same chunk text, same model, MRL-truncated to 1024 for both doc+query via the gateway), so the contradiction is real, not an index artifact.
- **This retires the plan's Phase 1 entirely.** The FP8/Matryoshka/warm-serve engineering it proposed would have been spent chasing a null. Net Phase-1 outcome = a rigorous negative + the incidentally-discovered BM25 fix (which is the actual value).

**Reframe of where value now lives** (all evidence-led):
1. **The BM25 fix** (already shipped `5d36dd0`) — restores production hybrid; likely the single biggest real improvement of the campaign, because hybrid (0.57–0.58) >> dense (0.48–0.50) and hybrid was silently off locally.
2. **Phase 3A citation layer** — now the most promising untested lever (keying ~89%, co-citation edges 100k→532k+). Test `hybrid_citation` vs `hybrid_sparse2` next.
3. Everything embedding-model-related (0.6B vs 4B vs 8B) is a dead end at current corpus/gold — consistent with "measurement/substrate-bound, not model-bound."

<!-- New entries appended below, newest-last, always timestamped. -->
