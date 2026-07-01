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

<!-- New entries appended below, newest-last, always timestamped. -->
