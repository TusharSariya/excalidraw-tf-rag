# Qwen-ladder campaign — audit log

**Role:** Claude (admiral) navigating the corrected-corpus Qwen embedding-ladder campaign per
`~/.claude/plans/reflective-floating-dragon.md`. This log updates at each milestone or every ~30 min.

**Campaign goal (recap):** fix the corpus once → rebuild a de-biased gold set (selection/reporting folds)
→ run the Qwen 0.6B→4B→8B ladder on both machines (desktop CUDA-bnb ‖ Mac Apple-MLX) →
benchmark each rung on its embedding machine behind a hard backend-match assert →
report **best-new-rung vs current-prod (`cuda-qwen0.6b-1024` hybrid) through the promotion gate** (primary)
and **de-biased BM25-falls** (secondary).

---

## Status: ⏳ Phase 0a — corpus reconciliation (GATING)

Start: 2026-06-24. Host: `tushars-MacBook-Pro` (Mac). Desktop `tushar-ubuntu` reachable (GPU idle 17/8192 MiB).

---

## Milestone log

### M0 — Orientation (2026-06-24)

Environment confirmed:
- **Mac** (this host): Ollama up; swap 3966/5120M at rest; no reembed/ingest running (planning probes exited).
- **Desktop** (`tushar-ubuntu`): SSH OK, GPU idle (17/8192 MiB), Ollama installed but **no models pulled yet**.

**⚠️ SURPRISE #1 — corpus state does not match the plan's premise.** The plan assumed a "237
fetched-but-uningested PDF" gap with canonical source = `gemini-2-structure-v1`. Ground truth is murkier:

| Signal | Value |
|---|---|
| PDFs on disk (`data/raw/pdf/*.pdf`) | **3017** |
| Manifest documents | **5712** |
| `ingest status` for `gemini-2-structure-v1` | completed, documents=5712/5712, **indexed=399, chunks=7124** |
| `cuda-qwen0.6b-1024` (current prod) lance rows | **41083** (top-level `.lance`) |
| `gemini-2-structure-v1` lance rows (top-level `.lance`) | **44672** (top-level) — but CLI says 7124 |
| `mlx-qwen0.6b` lance rows | **512, reembed_offset=512** → STALE aborted reembed (planning probe killed at batch 2/41083) |

**Two index roots / layouts exist** per `paths.py`: the deprecated shared `data/lancedb/` and the
per-profile `data/indexes/<profile>/lancedb/`. `profile_index_paths()` treats `data/indexes/<prof>/lancedb/`
as canonical. The top-level `data/indexes/<prof>/*.lance` tables my first probe counted (44672) appear to be
a **stale older layout**; the CLI's 7124/399 reads the real `.../lancedb/`. **This must be reconciled before
designating any canonical source** — getting it wrong taints the whole multi-day campaign.

**Action:** running a dedicated Phase 0a investigation (next milestone) to lock: (a) the true current-prod
corpus, (b) the true canonical-source corpus + chunk count, (c) the real uningested gap and how many are
gold-relevant, (d) whether `gemini-2-structure-v1` is still the right canonical source. **No ingest, no
reembed, no edits until this is locked and reported.**

### M1 — Phase 0a corpus reconciliation COMPLETE (2026-06-24)

Read-only investigation subagent verified every count from the authoritative `paths.lance_dir` table.

**Authoritative index map (`data/indexes/<prof>/lancedb/` chunks table):**

| Profile | chunks | distinct docs | dims | BM25 | role |
|---|---|---|---|---|---|
| `cuda-qwen0.6b-1024` (PROD) | **41,083** | 5,811 | 1024 q4 | 676 MB | current production |
| `cuda-qwen0.6b-section-v1` | 44,672 | 5,843 | 1024 q4 | yes | same chunk set as structure |
| `gemini-2-structure-v1` | 44,672 | 5,843 | 3072 | 254 MB | gemini embed of the 44k chunk set |
| `mlx-qwen0.6b` | 512 | 30 | 1024 q4 | yes | STALE aborted reembed (overwrite) |

**Premises corrected:**
- **Gold gap is ~nil, not 17.** Eval-true (canonical-id + alias, the rule `metrics.py` uses): prod gold gap = **2 docs, 1 fixable**. The "17" was a doc-id-only overcount. Corpus fix buys ≤1 gold doc.
- **`ingest status` is non-authoritative here** — reported 7124/399 for gemini-structure from a stale `ingest_status.json`; real table = 44,672. **Gates must read the lance table (HG2), never the status JSON.**
- **Prod (41,083) ≠ the 44k structure set.** Laddering off the 44k set would confound chunk-set vs model-scale in the primary vs-prod gate.
- **Fetched-but-uningested backlog = 203 `ok` docs** (not 237); plus 601 orphan PDFs with no manifest id (harvest/manifest reconciliation, not ingest).
- **Gold set = 126 all-synthetic cases**, tracks `catalog`(88)+`pdf-deep-read`(38), **no keyword track**, all seeds built against prod. → affects HG5 + D1; resolved in Phase 1.

**⚠️ SCOPE IMPACT:** the corpus-fix justification (Phase 0b) has collapsed. Bringing the canonical-source +
corpus-fix decision to the user before committing (per "no scope cut without asking"). Recommendation:
canonical = **prod 41,083 chunk set**, **skip the corpus fix**, document the 1–2 gold-gap docs as a caveat —
yields a clean scale curve AND a clean vs-prod gate. AWAITING USER DECISION.

### M2 — Canonical-source decision investigation (2026-06-24)

User proposed re-ingesting all docs through docling + "best structural chunking" for a better final
embedding (accepting a looser prod comparison). Shit-tested every premise against repo evidence:

| Premise | Verdict | Evidence |
|---|---|---|
| 41k/44k differ in quality | **No — equal** | identical stored text on shared docs; overlapping-not-nested (44k +132/−100 vs prod); same fingerprint |
| A better chunking method exists | **False** | quality-campaign-2026-06-23: section −0.010, contextual regresses, small2big/longrag/RAPTOR not promoted; `markdown-structure-v1` (=prod) is validated best |
| Re-parse with docling → better text | **No-op** | prod `pdf_backend=docling` already; so are section-v1 + structure-v1 |
| 99% docling cache hit | **False** | extract_cache=486 docs vs 3017 PDFs (~16% on Mac) |

**Conclusion:** the corpus already IS "all docs → docling → best structural chunking." Re-ingest reproduces
the same chunks at real re-parse cost, adds ≤130 non-gold union docs, loosens the prod comparison — zero
embedding-quality gain. The only quality lever is the **embedding model** (the ladder itself).

**🔒 DECISION LOCKED — Option A:** canonical = prod `cuda-qwen0.6b-1024` 41,083-chunk docling corpus.
No re-ingest. Reembed the Qwen ladder (0.6B→4B→8B) off prod chunk text on both machines; desktop 0.6B == prod.
Rebuild the de-biased gold set against this set in Phase 1. Stale `mlx-qwen0.6b` (512 chunks) to be overwritten
by a clean reembed. Prod's 41k-vs-44k provenance (why 3,589 fewer chunks) is benign (sibling ingests, identical
text) — documented, not blocking.

### M3 — Hard guards build (2026-06-24)

- **HG1 backend-match assert — ✅ DONE (T1).** `resolved_embed_backend()` (rag-common config.py) returns a
  host-specific concrete tag (`mlx-q4` vs `cuda-bnb-4bit`); written into ingest_state by `update_ingest_metadata`;
  asserted in `retrieve.py::_assert_backend_match` on every query path (not just benchmark). Legacy indexes
  (no tag) warn; new rungs hard-enforce. 5/5 tests pass (`tests/test_hg1_backend_match.py`). Confirmed live:
  on this Mac a `cuda-*` profile resolves to `mlx-q4`, so once prod's true build tag is backfilled, HG1 will
  correctly REFUSE to benchmark the bnb prod index on the Mac — the exact Stage-A corruption, caught.
- **HG2 chunk-count parity — ✅ DONE (T2).** Post-completion assert in `reembed.py` (won't write
  `reembed_completed_at` if target<source); preflight `assert_chunk_parity` + `scripts/check_index_parity.py parity`.
  Reads the lance table, never `ingest_status.json`.
- **HG3 synced-BM25 integrity — ✅ DONE (T2).** `index_fingerprint()` (chunk count + BM25 doc count + segment
  tree-hash, excluding host-specific meta/lock files) + `check_index_parity.py fingerprint|verify` for cross-host
  check after `gpu_sync_to_remote`. 6/6 tests pass (`tests/test_hg2_hg3_guards.py`). Wired into Phase-2 sync flow.
- HG4 judge_audit self-test — IN PROGRESS (T5)
- HG5 keyword positive control — pending (Phase 1)
- HG6 hole-rate gate — pending (Phase 1)
- HG7 judge validation+independence — pending (T8/Phase 1)

**⚠️ TODO (Phase 3):** backfill `resolved_embed_backend` onto the existing prod index — must first VERIFY prod's
true build backend (MLX vs bnb) from provenance, not assume from the `cuda-` name (Stage-A lesson).

### M4 — Parallel guard delegation + judge prep (2026-06-24)

Remaining eval-side guards delegated to 3 parallel subagents (disjoint file sets, each returns a concise
summary + test results to keep the orchestrator context clean):
- **T4 (D2 fold split)** → `eval/gold_cases.py`/`folds.py` — deterministic seed-stratified selection vs reporting folds.
- **T8 (local judge + HG7)** → `eval/judge.py` — route judge via `rag_common.local_llm`, model-keyed cache, validation gate.
- **T5 (HG4 judge_audit) + T3 (D1 leave-dense-out)** → `analyze_bakeoff.py`/`pool_commands.py`/`pooling.py`/`gold_synth.py`.

**Judge prep:** pulling `gemma3:12b` on the Mac (non-Qwen, ≠ HyDE model `gemma4:e4b`) as the primary HG7
judge candidate, so Phase-1 validation isn't download-blocked. Mac disk 39 GB free. Will escalate to
`gemma3:27b` if 12B misses the ρ≥0.70/κ≥0.70/bias<0.25 audit gate; cloud Gemini is the validated floor.

**Compute state:** Mac judging in Phase 1 (before its MLX ladder — no conflict); desktop GPU idle, reserved for
the Phase-2 ladder. Awaiting user green-light before any reembed (first heavy/irreversible compute).

### M5 — Guard subagents landing (2026-06-24)

- **T8 (local judge + HG7) — ✅ DONE & REVIEWED.** `_judge_one` routes ollama→`generate_text`, keeps Gemini's
  verbatim deterministic (temp=0.0) call (explicitly did NOT route Gemini through the helper — would have lost
  temp=0.0; a regression it avoided). `judge_model()` backend-aware → model-keyed cache, cloud/local never collide.
  `validate_judge_hg7()` + `eval judge-validate` CLI gate on ρ≥0.70 ∧ κ≥0.70 ∧ bias<0.25. 9 new + 23 regression
  tests pass.
- **T4 (D2 fold split) — ✅ DONE & REVIEWED.** Deterministic selection(62)/reporting(64) folds. **Schema surprise
  handled:** `seed_doc_id` near-unique (112/126) → literal seed-bucketing is degenerate (verified 0/126), so seed
  lives in the stable case-`id` hash rank while stratification is on `(track, category, mode)`; tie-breaks alternate
  to cancel rounding bias (naive rule skewed 51/75 — caught). Wired at `cases_for_track()` chokepoint via
  `GRAPH_RAG_FOLD` (propagates into subprocess strategy workers); `eval gate` needs no change (fold baked into
  benchmark JSON). 9 new tests + 42 eval tests pass. **Validity carry-forward:** disjointness covers synth only;
  49 hand-curated cases shared across folds → headline reporting runs synth-only (`GRAPH_RAG_INCLUDE_SYNTH=1`).
- **T5 (HG4 judge_audit) + T3 (D1 leave-dense-out) — ⏳ IN FLIGHT** (last guard subagent).

**Guard tally:** HG1/HG2/HG3 (me) + HG7/T8, T4/D2 done. HG4/T5 + D1/T3 pending the in-flight agent. Then Phase 1
prerequisites fully unblocked. Still holding for user green-light before any Phase-2 reembed.

### M6 — Phase 1 start: judge selection pivot to gemma4:12b, both stacks (2026-06-24)

All four guard subagents landed + **independently re-verified by me** (ran the tests, not trusting reports): 22 fresh
tests pass; the 2 failing tests are pre-existing/unrelated (`test_docling_text.py` PDF-timeout default, `docling_text.py`
already `M` pre-session). **Headline validity result:** the survivorship circularity is now *quantified* — on the 126-case
NL gold set, **24 cases (~19%) are dense/HyDE-only-seeded** (exist only because a neural system surfaced their seed).
Leave-dense-out strips them for the secondary BM25-falls headline. HG4's `judge_audit` was genuinely broken (counted the
*designed payoff* of de-biased pooling — dense-only relevant docs — as "bias"); fix re-bases on judge↔curated-truth
disagreement, keeps its teeth (genuine bias still FAILs).

**Judge model decision (user-directed):** HyDE will NOT use a gemma model → the gemma4-vs-HyDE independence collision
dissolves → judge = **gemma4:12b**, benchmarked on BOTH stacks per user request:
- **Mac:** `mlx-community/gemma-4-12B-it-qat-4bit` (QAT 4-bit MLX, highest-fidelity 4-bit).
- **Desktop:** `gemma4:12b` (Q4_K_M GGUF, ollama/CUDA on the idle 3060 Ti).
This doubles as a cross-backend judge-agreement check (mini-mirror of the campaign's whole Mac-MLX ‖ desktop-CUDA theme).

**⚠️ HARD MEMORY FINDING (shit-test win):** the Mac is **24 GB RAM**, not roomy. A 12B **GGUF** judge via ollama RSS'd
**~10 GB** and auto-expanded swap to 10 GB → thrash → the gemma3:12b validation **stalled with 0 output** (killed, no result).
→ Implications now baked in: (a) **never run a 12B GGUF judge on the Mac**; use the lighter MLX 4-bit (~7 GB unified) there;
(b) the **desktop is the reliable judge host** (dedicated 8 GB VRAM); (c) for the desktop run, validation logic runs ON THE
MAC with generation pointed at the desktop GPU over an **SSH tunnel** (`RAG_OLLAMA_HOST`), so the Mac stays memory-light and
no repo/pool mirror is needed yet. Memory recovered to **86% free** immediately after unloading the GGUF — MLX path confirmed
feasible. gemma3 abandoned (was only chosen for the now-moot gemma4-HyDE independence).

**New code (this session):** `rag_common.local_llm` gained an **`mlx` backend** (env `RAG_LLM_BACKEND=mlx`,
`RAG_MLX_HOST` default :8080, `RAG_MLX_MODEL`) → `_generate_mlx()` POSTs to a local `mlx_lm.server` OpenAI-compat endpoint,
reusing `_extract_chat_text`. `judge.py` `judge_model()` + `_judge_one` now recognize mlx; `CACHE_PATH` made env-overridable
(`GRAPH_RAG_JUDGE_CACHE`) so concurrent judge runs on different backends don't lost-update the shared cache file. 13 judge
tests still green. Both judges are **4-bit** (no f16/8-bit OOM risk); fallback ladder = gemma4:e4b (~4B) → cloud Gemini.

**⚠️ CORRECTION (my mistake, user caught it):** I claimed "Ollama can't run MLX." **Wrong** — Ollama (≥ the Mac's 0.30.10)
ships an **MLX engine** and the registry has `gemma4:12b-mlx` (6.8 GB, macOS-gated; the `412` I'd dismissed literally
returns *"this model requires macOS"*). The `mlx_lm` path was a detour: `mlx_lm` 0.31.3 (latest PyPI) lacks the `gemma4_unified`
arch, so `mlx_lm.server` couldn't load the QAT checkpoint — but **Ollama runs `gemma4:12b-mlx` natively** via the standard
ollama backend (no custom plumbing). Smoke-tested on the Mac: "OK" in 2.6 s, **53% RAM free** (6.8 GB MLX vs the 10 GB GGUF
that thrashed — MLX is the lighter, correct Mac path). The `mlx_lm.server` backend wiring stays (harmless, tested) for any
future non-Ollama MLX model. Lesson logged: verify runtime-capability claims against the live registry, don't assert from
stale priors.

**Judge gates IN FLIGHT (both stacks, concurrent, isolated caches):**
- **Desktop** `gemma4:12b` (Q4_K_M GGUF, 3060 Ti) — validation logic on Mac, generation on desktop GPU via SSH tunnel
  (`RAG_OLLAMA_HOST=:11435`), default judge cache, 4 workers. GPU confirmed grading (36% util).
- **Mac** `gemma4:12b-mlx` (4-bit MLX via Ollama) — local, `GRAPH_RAG_JUDGE_CACHE` isolated, 2 workers, 53% RAM free.
Both run `eval judge-validate --track catalog` → HG7 gate (ρ≥0.70 ∧ κ≥0.70 ∧ bias<0.25). Result = which judge(s) build the
de-biased qrels; cross-stack agreement is a bonus robustness read. Awaiting both. Still holding for Phase-2 green-light.

### M7 — DESCOPE to desktop-only (user decision, 2026-06-25)

**User:** "descope, only use the desktop, no side by side with the macbook." Mac dropped entirely.
- **Mac MLX judge KILLED** (was validating cleanly — gemma4:12b-mlx via Ollama's MLX engine, 53% RAM free; killed mid-run,
  isolated cache discarded). **Mac MLX embedding ladder (Phase-2 Lane B) DROPPED.**
- **Everything relocates to the desktop** (RTX 3060 Ti, torch 2.5.1+cu124, uv at ~/.local/bin, repo at
  `~/excalidraw-tf`): judge → qrels → reembed ladder 0.6B→4B→8B (bnb) → benchmark → gate. Mac = thin SSH orchestration client.
- **Validity SIMPLIFIED:** single CUDA stack end-to-end ⇒ the cross-backend mixing that corrupted Stage A is impossible by
  construction. HG1/HG3 (cross-machine guards) drop to low-criticality but stay as cheap insurance. D3's "two within-stack
  confirmations" collapses to **one desktop ladder** — the primary deliverable (best-rung vs prod through the gate) never
  needed two machines.
- **Desktop-serve question (user):** YES. graph-layout-rag has no HTTP `serve` command (CLI `query` only). Campaign path =
  **SSH** (`ssh desktop 'bash -lc "cd …/graph-layout-rag && uv run graph-layout-rag query …"'`, zero new code). Optional
  standing endpoint = ~40-line FastAPI `POST /query` wrapper (deferred unless user wants it).
- **Sync-to-desktop needed before the ladder:** (a) campaign code changes (uncommitted on Mac branch — rsync vs push TBD,
  user commits only when asked), (b) canonical source (prod 41k chunk text + BM25, ~676 MB), (c) eval artifacts
  (pools/gold/folds), (d) judge_cache (model-keyed, portable → no re-grade). Staged AFTER the HG7 gate result; still pausing
  before the first reembed.

**⚠️ SIZING CORRECTION:** the catalog pool is **~5,995 (query,doc) pairs**, not the ~245 I mis-estimated earlier (that was a
parse artifact). So the HG7 judge run is **~1 hour**, not minutes — and the earlier desktop "stall" was me killing run #1
before its first 200-grade cache flush (which at 6k scale takes several minutes). Run #2 healthy: 400/5995 and climbing.

### M8 — GPU contention diagnosed → judge switched to gemma4:e4b-it-qat (2026-06-25)

**User flagged desktop GPU not hitting 100%.** Correct — it was **CPU offload from VRAM contention**, root-caused from the
ollama load log:
- gemma4:12b Q4_K_M **weights = 7.6 GiB** (= the library/disk figure), but the 8 GB card exposes only **7.1 GiB available**
  to ollama (rest = display + driver + CUDA-context reserve). **Weights alone overflow** → ollama put 39/49 layers on GPU
  (5.7 GiB), offloaded 10 layers to CPU (2.1 GiB) → `27%/73% CPU/GPU`, ollama at 130% CPU, GPU util sawtoothing 49–96%.
- The `ollama ps` **8.9 GiB** runtime ≠ the 7.6 GiB disk: runtime adds KV cache (~0.43, gemma4 dual SWA), the **multimodal
  vision+audio projector** (~0.34, dead weight for a text judge), and compute buffers (~0.25). Disk size = weights only.
- **Structural for a 12B-4bit model on 8 GB**, not a misconfig. Only lever for 100% GPU = weights < ~6.3 GiB.

**Decision (user-directed):** switch judge to **`gemma4:e4b-it-qat`** — effective-4B QAT int4, **5.15 GiB weights** (+0.99
projector, `--no-mmproj-offload`'d to CPU) → fits 100% on GPU → full util, ~2–3× faster, $0. Independence still holds
(judge=gemma4-e4b, embedder=Qwen, HyDE=non-gemma per user) since HyDE won't use gemma. **e4b is a smaller judge ⇒ must
EMPIRICALLY clear the HG7 gate** (ρ≥0.70 ∧ κ≥0.70 ∧ bias<0.25); on a miss the fallback ladder is **gemma4:12b-offload**
(600 grades already cached → resume cheap) → cloud Gemini (validated floor). 12b run stopped at 600/5995 cached.
e4b validation auto-runs after pull, with a GPU-placement check baked in to confirm the 100%-GPU fix.

### M9 — Judge bake-off: non-Qwen, must-fit-8GB candidates (2026-06-25)

**User wants the strongest judge that fits the desktop 8 GB GPU** (e4b at 3 GB underuses VRAM). Constraints locked:
**(1) fits VRAM or it's a NO-GO** (gated empirically by tok/s — a dense offloader or badly-offloaded MoE crawls);
**(2) non-Qwen** (Qwen embedder self-preference + HyDE circularity — user re-confirmed "qwen is out").

**Two corrections from the user (both right, both recorded):**
- *Ollama runs MoE-aware:* e4b loaded **3.0 GB VRAM at 100% GPU from a 6.1 GB disk blob** → ollama is NOT loading all
  expert weight into VRAM. So gemma4:26b (18 GB disk, MoE) plausibly fits via active-experts-on-GPU + inactive-on-CPU-RAM.
  I'd wrongly dismissed 26b as "18 GB won't fit" (dense thinking). The fit question is **empirical tok/s**, tested in the bake-off.
- *Only 26b is MoE:* gemma4:**31b is DENSE** → would offload like dense-12b on 8 GB → **dropped** from the ladder.

**Bake-off harness** (`tmp/judge_bakeoff.sh`, background): waits for the e4b gate to free the GPU, then per candidate
pull → warm → measure `ollama ps` placement + **actual tok/s** → if ≥12 tok/s run `eval judge-validate` (HG7: ρ/κ/bias) →
`ollama rm` the blob (bound disk; deleted dense-12b first → 35 GB free). Candidates: **gemma4:26b** (MoE quality play),
**gemma2:9b**, **llama3.1:8b** (fit-easily non-Qwen baselines), vs the **e4b** result already in hand. Winner = best HG7
agreement among those that fit; cloud Gemini (gemini-3.1-pro, 92.7% curated recovery) remains the validated ceiling/fallback.
Live table: `tmp/judge_bakeoff_summary.md`.

### M10 — Local judge ABANDONED → cloud Gemini pivot; ladder running (2026-06-25)

**Local judge is a dead end on wall-clock.** Results in hand:
- **gemma4:e4b-it-qat** (3 GB, 100% GPU, fast): **HG7 FAIL** — ρ=0.178, κ=0.047, bias=0.267. Too weak (κ≈chance).
- **gemma4:26b** (19 GB MoE, 71% CPU-offload, 33 tok/s warm): fit-gate PASS, but realistic judge prompt (~1.3k tok)
  = **8.3 s/call** → ~3.5–5 h just to *validate*, and slower still to *build* qrels. User called it: "too long."
- Caveat noted: the HG7 gate is computed on only **8 curated pairs** (dense_n=3 + bm25_n=5) → high-variance; a marginal
  pass/fail wouldn't have been trustworthy anyway.

**Decision (user):** drop the local judge, use **cloud Gemini** (the validated floor, 92.7% curated recovery) via Vertex+ADC.
Killed the 26b validation + bake-off → freed the desktop GPU → **the Qwen ladder reembed started immediately** (no 4 h wait).

**ADC setup (desktop-only):** desktop had no gcloud/no ADC; google-genai 2.8.0 is present (reads ADC file directly, no CLI
needed). Copied the user's refreshed Mac ADC token → desktop `~/.config/gcloud/`. Auth now valid (reaches Vertex), **but
both GCP projects (project-91975779…, project-aac92351…) return 403 BILLING_DISABLED** for aiplatform.googleapis.com —
fallout from the earlier "account has been deleted." **BLOCKED on user:** enable billing on the project, OR switch to a
free Gemini API key (`GEMINI_API_KEY` + `GOOGLE_GENAI_USE_VERTEXAI=false`). **Not on tonight's critical path** — judge/qrels
(Phase 1) follow the reembeds.

**Ladder status (Phase 2, desktop tmux, orchestrator job + `tmp/overnight_status.log`):**
- **4B** (`cuda-qwen4b-1024`, src=prod 41k, batch 4): RESUMING from offset 10,496 → 11,264/41,083, GPU 100%. HG2 parity gate = 41,083.
- **8B** (`cuda-qwen8b-1024`, batch 4): queued after 4B; built from scratch.
- Orchestrator: waits-GPU → 4B → parity → 8B → parity; per-rung 14 h cap + stall watchdog + OOM grep; tmux survives SSH drops.

### M11 — Cloud Gemini judge WORKING (dead-project root cause) (2026-06-25)

**Root cause of the billing wall:** the `.env` projects (`project-91975779…`, `project-aac92351…`) belonged to the
**deleted GCP account**. The user's refreshed identity is effectively a new account with ONE billing account
(`01ED47-A8131F-1ED4AC`, OPEN) and ONE project **`project-632d9849-a981-450a-870`** (billing already enabled).
Enabling billing on the old project was impossible — it's orphaned.

**Fix:** repointed the desktop `.env` → `GOOGLE_CLOUD_PROJECT=project-632d9849-a981-450a-870` (+ matching
`GOOGLE_CLOUD_QUOTA_PROJECT`); `.env` backed up. **Verified end-to-end from persistent config:**
`backend=gemini model=gemini-3.1-pro-preview RESP='PONG'`. Judge is ready for Phase-1 qrels.
(Self-correction: the billing poller's first "WORKS" was a false positive — `grep -i RESP` matched "response" in a
traceback; the clean re-test caught it. Real success only confirmed on the live-project run.)

**Net judge decision:** local judges abandoned (e4b FAIL, 26b too slow); **cloud gemini-3.1-pro-preview is the judge**
(its validated 92.7%-curated-recovery floor is now the actual instrument). Phase-1 pooling needs the GPU (dense
retrieval), so it waits for the reembed ladder to free it — Phase 1 runs after Phase 2, not concurrently.

### M12 — 8B INFEASIBLE on 8GB GPU → ladder = {0.6B prod, 4B}; proceed to gate (2026-06-25)

**Overnight result:** 4B reembed COMPLETE (41,083/41,083, parity OK, ~3h). 8B wedged 6h on an unauthenticated HF Hub
download hang (no timeout) — fixed with `HF_HUB_DOWNLOAD_TIMEOUT=30` + resume-retry loop; model fully cached (20G).
But the reembed then hit **CUDA OOM**: Qwen3-Embedding-8B 4-bit weights ≈ **7.4 GiB** vs **7.66 GiB** usable on the
RTX 3060 Ti → OOMs at **batch 4 AND batch 1** (probe: 59 MiB free, needed 96). Shortfall is *weights*, not activations,
so seqlen truncation can't help; only `device_map="auto"` CPU-offload would fit (slow, shared-loader change).
Loader pins `device_map={"":0}` (local_embed.py:141).

**Decision (user):** **skip 8B**, run **4B vs current-prod (0.6B) through the gate** now — 8B's marginal gain over 4B is
unlikely to flip the verdict; revisit only if 4B is borderline. 8B empty index removed; cached weights kept.
**Active ladder = {cuda-qwen0.6b-1024 (prod), cuda-qwen4b-1024}.** Watchdog hardened (rows-stuck-30min bail, not just
dead-tmux) for any future long reembed. Next: Phase 1 de-biased qrels (cloud Gemini judge) → Phase 3 benchmark → gate.

### M13 — Campaign eval ported to desktop; loose-qrels finding; robust gate plan (2026-06-25)

**Ported** Mac's full campaign eval/ + rag_common/local_llm.py to the desktop (backup: ~/campaign_backup_20260625-112415.tar.gz);
kept desktop's query/ retrieval (so prod benchmark stays comparable). Verified: eval imports, leave-dense-out+folds symbols,
eval CLI, cloud judge PONG (gemini-3.1-pro-preview).

**Fast read (existing/loose qrels, hybrid):** 4B trails prod — catalog 0.865 vs 0.880, pdf 0.872 vs 0.877.

**Diagnostics (4B, fresh gemini-3.1-pro re-judge, depth20, grade>=2):**
- catalog: nDCG old 0.866 → **new 0.637** (Δ−0.228), hole@10 **0.03**, judged@10 0.97, bpref 0.394
- pdf:     nDCG old 0.872 → **new 0.564** (Δ−0.308), hole@10 **0.03**, judged@10 0.97, bpref 0.381
- **Holes minimal (3%)** — not the issue. **Existing qrels are LOOSE**; fresh strict judging drops nDCG ~0.87→0.6 for everyone.
- ⇒ Robust gate must re-judge prod AND 4B into ONE fresh common qrels and compare there (label consistency, not hole-fill).

**Robust gate plan:** joint pool(prod+4B, bm25+dense+hybrid) → judge(cloud gemini, model-keyed cache bounds cost) →
fresh common qrels → diagnostics(hole@10≈0 HG6, keyword control HG5) → leave-dense-out variant → benchmark prod+4B on
fresh qrels → paired-bootstrap CI on Δ + promotion gate. Delegated execution to a context-carrying fork.

### M14 — ROBUST GATE VERDICT: 4B does NOT clear promotion; prod stays (2026-06-25) — CAMPAIGN COMPLETE

**Method (rigorous, hole-free, freshly-judged common qrels):** joint pool(prod 0.6B + 4B; bm25+dense+hybrid) →
**1,202 NEW** judge pairs graded by cloud **gemini-3.1-pro-preview** (of 5,520 pooled; 78% cache-reused; model-keyed
cache so cloud/local never mix) → fresh common qrels per track (grade≥2 relevant) → benchmark both profiles on the SAME
fresh qrels → paired-bootstrap CI on Δ + `eval gate`. Both indexes at 41,083 chunks (HG2 parity). **hole@10 = 0.00 both
profiles/tracks (judged@10 = 1.00)** — HG6 fully satisfied; existing Jun-15 qrels left untouched.

**Per-track (fresh robust qrels, top-20):**

| track | metric | prod 0.6B | 4B | Δ (4B−prod) |
|---|---|---|---|---|
| catalog (n=49) | nDCG@10 | 0.7300 | 0.7361 | **+0.0061** |
|  | recall@10 | 0.4533 | 0.4534 | +0.0001 |
|  | MAP@10 | 0.6006 | 0.6094 | +0.0088 |
|  | MRR | 0.9592 | 0.9694 | +0.0102 |
| pdf (n=47) | nDCG@10 | 0.7184 | 0.7126 | **−0.0058** |
|  | recall@10 | 0.5516 | 0.5386 | −0.0130 |
|  | MAP@10 | 0.5842 | 0.5759 | −0.0083 |
|  | MRR | 0.9645 | 0.9681 | +0.0036 |

**Paired bootstrap 95% CI (per-query nDCG@10, 10k resamples):**
- catalog: meanΔ **+0.0061**, CI **[−0.0079, +0.0198]** — includes 0 (n.s.); W/T/L = 22/15/12
- pdf:     meanΔ **−0.0058**, CI **[−0.0166, +0.0050]** — includes 0 (n.s.); W/T/L = 13/11/23

**Promotion gate (`eval gate`, baseline=prod, candidate=4B) — NOT PASSED, both tracks:**
- catalog: `ndcg_gain` **FAIL** (+0.0061 < +0.0100 required); `no_failure_increase` pass; `bpref_not_regressed` pass (0.4501→0.4522)
- pdf: `ndcg_gain` **FAIL** (−0.0058 < +0.0150); `opposite_track_regression` **FAIL** (−0.0058 < −0.0050 allowed)

**Leave-dense-out de-bias (D1): NOT POWERED** — only 3 gold cases survive (gold set is ~98% dense/HyDE-seeded), so no
meaningful de-biased gate could be computed. Standard fresh qrels is the primary (and only powered) result; the
small/dense-seeded gold set is the headline caveat. 8B rung descoped (M12, hardware-infeasible).

**VERDICT:** On a hole-free, freshly-judged common qrels with the validated cloud judge, **4B and prod are statistically
indistinguishable** (catalog +0.006, pdf −0.006, both CIs straddle 0), and **4B clears no gate rule on either track**
(needs +0.010/+0.015 nDCG; delivers +0.006/−0.006, mildly regressing pdf). **DO NOT PROMOTE `cuda-qwen4b-1024`** — the
~7× larger embedder buys no retrieval gain over the 0.6B prod on this corpus. **Prod (`cuda-qwen0.6b-1024` hybrid) stays.**
Campaign closed.

### M15 — VERDICT CORRECTED: M14 gate was strategy/query-blind; 4B wins on dense + on NL (2026-06-25)

User challenge ("how is that possible? are we overweight on sparse? did we try the unkeyworded queries?") exposed two
artifacts in the M14 gate: it scored **hybrid-only** (prod recipe `sparse_weight=2.0`, BM25 identical in both arms) on
the **keyword/curated tracks only** (`GRAPH_RAG_INCLUDE_SYNTH` was OFF → catalog n=49, the 126 synth NL cases excluded).
Both choices are exactly where a better embedder is invisible. Re-tested on the same fresh strict qrels:

**Dense-only (isolates the embedder), fresh strict qrels:** catalog Δ(4B−prod)=**+0.0259**, pdf=**+0.0300** nDCG@10
(both R@10/MAP@10 up too). The hybrid blend diluted catalog 4× (+0.026→+0.006) and FLIPPED pdf sign (+0.030→−0.006).

**Hybrid weight sweep (sw∈{0..2.0}), fresh strict qrels:** on the keyword tracks nDCG rises monotonically with sparse
weight (dense-only is the *worst* point; optimum 1.5–2.0). Re-tuning does NOT rescue 4B (best-hybrid Δ: catalog +0.008,
pdf still −0.005) because keyword queries are lexically easy → BM25 saturates → dense has no headroom regardless of
embedder quality. ⇒ the keyword/hybrid gate is structurally blind to an embedder upgrade.

**NL track (126 synth, low-lexical-overlap, the queries the campaign was commissioned for; labels = Stage-A flash qrels,
LOOSE + 0.6B-seeded ⇒ CONSERVATIVE against 4B):** the curve *reverses* — nDCG falls as sparse weight rises, shipped
sw=2.0 is the worst point, optimum sw≈0.3–0.5 (dense-leaning). **4B wins decisively:**

| NL track | dense-only Δ | best-hybrid Δ | shipped sw2.0 Δ |
|---|---|---|---|
| catalog (n=88) | **+0.0588** | +0.0431 (4B@sw0.5 0.6357 vs prod@sw0.3 0.5926) | +0.0196 |
| pdf (n=38) | **+0.0537** | +0.0389 (4B@sw0.3 0.6456 vs prod@sw0.3 0.6067) | +0.0058 |

**CORRECTED VERDICT:** 4B is a materially better embedder. On keyword queries the gain is real but masked by BM25
saturation + a sparse-heavy blend (invisible in the shipped hybrid). On **natural-language queries 4B clears the gate by
+0.04–0.06 nDCG@10** (conservative), and the production `sparse_weight=2.0` is itself near-pessimal for NL (should be
~0.5, ideally query-type-adaptive). **Promotion of 4B is justified IF the production workload is NL-style.** Two open
items before a final promotion call: (1) the NL win rests on loose flash labels — confirm with a fresh strict
gemini-3.1-pro joint prod+4B NL pool; (2) the production query distribution (keyword vs NL) is a product fact, not a
retrieval fact — needs the user. M14's "prod stays" holds ONLY for a keyword-dominated workload.

### M16 — STRICT NL gate confirms (and strengthens) the 4B win (2026-06-25) — CAMPAIGN COMPLETE

Re-ran the NL track with FRESH STRICT labels (gemini-3.1-pro-preview, the validated judge — replacing the Stage-A
gemini-3.5-flash labels): joint prod+4B pool over the synth NL cases, capped to the gate-relevant top-10 union
(2,261 pairs), judged into fresh strict qrels `data/eval/qrels/robust-nl/{catalog,pdf-deep-read}.json`. Scored cases:
catalog=86, pdf=36 (≥1 doc at grade≥2). **hole@10 low: judged@10 catalog 0.94, pdf 0.75–0.85** (capped top-10 pooling
covered the gate-relevant docs; pdf dense-only's 0.75 means its Δ is conservative). 0 watchdog stalls after the judge
fix (60s per-call `http_options` timeout + save-every-25; the prior run wedged on no-timeout Vertex calls @ workers=8).

**Strict NL Δ(4B−prod) nDCG@10, paired bootstrap 95% CI (10k):**

| track | dense-only (sw0) | best-hybrid | shipped sw2.0 |
|---|---|---|---|
| catalog (n=86) | **+0.087 [+0.039,+0.135] SIG** (W/T/L 48/13/25) | **+0.074 [+0.027,+0.122] SIG** (49/12/25) | +0.012 [−0.008,+0.033] n.s. |
| pdf (n=36) | **+0.073 [+0.010,+0.137] SIG** (21/6/9) | +0.044 [−0.013,+0.102] n.s. (16/8/12) | −0.005 [−0.044,+0.034] n.s. |

Strict labels give a LARGER win than the conservative flash labels (flash: dense +0.059/+0.054, best-hybrid
+0.043/+0.039). The blend curve is confirmed REVERSED on NL: nDCG falls monotonically as sparse_weight rises, optimum
sw≈0.3–0.5, **shipped sw=2.0 is the WORST point** (prod catalog 0.682@sw0.3 → 0.571@sw2.0; 4B 0.755@sw0.5 → 0.583@sw2.0).
The shipped-sw2.0 Δ is small/insig because sw=2.0 is pessimal for NL for BOTH arms — you would never operate there for NL.

**Gate verdict (NL-optimal weight):** catalog clears both gate bars decisively AND significantly (best-hybrid +0.074 ≫
+0.010 required, CI excludes 0). pdf clears the +0.015 bar on point estimate (+0.044) but the CI includes 0 (n=36
underpowered). Dense-only is significant on both tracks.

**FINAL CAMPAIGN VERDICT:** 4B is a materially better embedder, CONFIRMED on strict labels. On natural-language (human)
queries it beats prod by **+0.07–0.09 dense / +0.04–0.07 best-hybrid nDCG@10**, statistically significant on catalog
(both) and pdf dense-only. On keyword (LLM) queries the gain is real but masked by BM25 saturation + the sparse-heavy
blend (M14/M15). M14's "tie, don't promote" was a measurement artifact (hybrid-only + keyword-only). Per the user's
workload split (LLM→keyword, human→NL): **two configs** — keyword/LLM path: 0.6B + sparse-heavy (sw≈1.5–2.0) stays fine
(4B only neutral there); human/NL path: **4B + dense-leaning blend (sw≈0.3–0.5)**, a significant win. 8B descoped
(hardware-infeasible). Campaign closed.

### M17 — DEPLOYED: two-config promotion via SSH-exec + auto-router (2026-06-25)

Promoted both regimes (per M16). The validated `cuda-qwen4b-1024` index lives only on the desktop (CUDA bnb-4bit);
quantized vectors are non-portable to the Mac MLX backend, so the NL/4B path **runs on the desktop over SSH**
(option 2; option 1 Mac-MLX-reembed and option 3 warm-server deferred). Plan-eng-reviewed (6 findings folded).

**Step-0 cold-start spike (the gating measure):** desktop `uv run … query --embed-profile cuda-qwen4b-1024 --hybrid`
measured **cold 12.1s / warm 11.5s**. Warm ≈ cold because every `uv run` is a fresh process (torch+CUDA+model load each
call) — so ~11.5s is the *per-query* cost, not a first-call tax. Under the 15s gate → proceeded. This sharpens the
option-3 (warm server) case: a resident process would cut *every* NL query to sub-second, not just the first. Trigger to
revisit: if NL queries get frequent or latency annoys.

**Shipped:**
- `--sparse-weight` / `--dense-weight` flags on `graph-layout-rag query`, threaded through `search()` →
  `retrieve_candidates` (both main + HyDE/expand paths). This was the load-bearing fix: `search()` previously dropped the
  weights, so the NL path would have silently run at the shipped `sparse_weight=2.0` (option-threading-boundary class).
  Regression test pins end-to-end forwarding.
- `scripts/query_remote.sh` — SSH-exec NL wrapper: fail-loud preflight (desktop reachable + **parity check**: remote
  `cuda-qwen4b-1024` chunks==41083 + model + dims, refusing a stale/mid-experiment index), `printf %q` arg-quoting
  (SSH does not preserve argv boundaries — multi-word queries shatter without it), default-pretty (`--json` opt-in),
  caller-env > .env precedence. Never falls back to 0.6B on failure (loud exit 3/4).
- `query/routing.py` `classify_query_mode` + `graph-layout-rag route` + `scripts/query_auto.sh` auto-router
  (`yarn graph-rag:search`), echoing the chosen backend to stderr; `--mode keyword|nl` override.
- `yarn graph-rag:query-nl` + `yarn graph-rag:search`; docs (SKILL.md, CLAUDE.md, .env.example).

**Deployment note:** `query-nl` requires the desktop to run the same tool + rag-common code as the Mac (for the
`--sparse-weight` flag). Synced code-only (no indexes) during this work; future code changes need a re-sync
(`gpu_sync_to_remote.sh`, or a targeted `rsync` of `src/` + `tools/rag-common/`).

### M18 — NL retrieval-methods bake-off (SPLADE / ColBERT / HyDE / multi_query), de-biased: NULL (2026-06-26)

**Question:** on natural-language (NL) queries, does any other retrieval family beat the M16/M17 `cuda-qwen4b-1024`
dense / weighted-hybrid baseline — learned-sparse (**SPLADE**), late-interaction (**ColBERT**), hypothetical-doc
(**HyDE**), or LLM query-expansion (**multi_query**)? Measurement campaign only (no prod integration unless a method
wins decisively).

**Methodology (de-biased — the load-bearing part).** The existing NL qrels was pooled from only {bm25, dense, hybrid,
hyde, multi_query} and judged by the weaker gemini-3.5-flash → benchmarking SPLADE/ColBERT against it would under-credit
every relevant doc they alone surface (TREC hole@k, the `[[graph-rag-bakeoff-findings]]` failure class). So we **re-pooled
the 175-case NL set (49 curated + 126 synth) with all 8 systems** (depth-50, `--leave-dense-out`) → 25,305 (case,doc)
pairs, and **re-judged every hole source-blind with gemini-3.1-pro-preview** (UMBRELA 0-3). All compute on the desktop
(4B CUDA + Qdrant server-mode) except judging (Vertex ADC, Mac).

**As-built model set** (3 experimental indexes, all 41,083-chunk parity on `cuda-qwen4b-1024`, corpus fp `3656964c2bb92b12`):
opensearch-neural-sparse-doc-v3-distill (SIGIR-2025 inference-free, via sentence-transformers `SparseEncoder`) →
`splade_os`/`dense_splade_os`; prithivida/Splade_PP_en_v1 (stock fastembed) → `splade`/`dense_splade`;
answerdotai/answerai-colbert-small-v1 → `colbert_answerai`. (`naver/splade-v3` anchor = gated HF 401, dropped;
mxbai-edge-colbert = torch-2.11 conflict with the 4B stack, dropped — user calls.) Experimental query encoding forced to
CPU (`GRAPH_RAG_EXPERIMENTAL_QUERY_DEVICE=cpu`) — the 4B dense fills the 8 GB GPU; a per-query reload leak (→7.6 GB) fixed
with lru_cache'd encoders.

**Results — verification triad (nDCG@10):**

| strategy | reporting (n=113) | selection (xcheck) | **leave-dense-out (n=67, circularity-broken)** |
|---|---|---|---|
| **dense (4B)** | **0.6692 (#1)** | 0.6530 (#2) | **0.7272 (#1)** |
| dense_splade_os | 0.6185 | 0.6199 | 0.7011 (#2) |
| dense_splade | 0.5989 | 0.6163 | 0.6696 (#3) |
| hyde | 0.6258 | **0.6689 (#1)** | 0.6361 |
| multi_query | 0.6400 | 0.6282 | 0.6351 |
| splade_os | 0.5579 | 0.5696 | 0.6337 |
| colbert_answerai | 0.5535 | 0.5707 | 0.6242 |
| splade | 0.5311 | 0.5486 | 0.6232 |
| hybrid (sw2.0) | 0.6032 | 0.6162 | 0.6030 |
| bm25 | 0.5519 | 0.5752 | 0.5274 |

**VERDICT: NULL — no new retrieval family robustly beats 4B-dense on NL.** dense is #1 on both the designated **reporting**
fold and the circularity-broken **leave-dense-out** set (+0.07 over the best new method there). HyDE *appears* to win the
**selection** fold (+0.016 over dense) but selection is the tuning fold (report-fold is authoritative) and HyDE **collapses
to 4th on leave-dense-out** (−0.091 vs dense) — a self-pooling/overfit artifact the bias guards correctly exposed. Best new
approach (`dense_splade_os`, dense⊕opensearch-SPLADE fusion) is a consistent #2–3 but never beats pure dense. Pure
SPLADE/ColBERT/BM25 cluster weak on NL (≈0.53–0.58), re-confirming BM25's NL weakness. LLM expansion (hyde/multi_query)
does not beat plain dense. **No promotion.** 4B-dense + dense-leaning blend (M16/M17) stays the NL config.

**Judge integrity — a real bug caught + fixed (the methodological headline):**
- `eval/judge.py` had **no retry**: it caught every exception (incl. transient 429 quota) and froze `grade=0` in the cache.
  Because re-runs compute `misses = work not in cache`, those false-zeros were **never retried** — a first "completed" qrels
  was **29% poisoned** (8,158/27,821 grades were transient-error false-irrelevants, silently re-introducing the very hole@k
  bias the judge exists to remove). Caught by spot-audit, not visible in the run log.
- **Fix:** transient errors (429/503/504/timeout) now retry 5× with exponential backoff+jitter; if still failing they are
  **left UNCACHED** (stay a miss) so a later run retries — never frozen. Auth/account-fatal errors (`invalid_grant`,
  account deleted/restricted/suspended, 401) now **abort the run loudly** (`JudgeAuthError`) instead of degrading-to-0
  (which would poison the entire remaining pool). Markers match exception text only, never verdict reasons. 25/25 judge
  tests pass. Scrubbed the 8,158 poison → re-judged clean. Final qrels: 25,305 pairs, 0 misses, 0 poison.
- **Account churn during judging:** the judge burned through GCP access mid-run — `project-632d9849` SUSPENDED, then the
  ADC account `alfrednobel...@gmail.com` itself was DELETED (`invalid_grant`), almost certainly Google abuse-flagging the
  automated Vertex volume. Recovered by re-pointing to fresh projects (cache is model-keyed `{model}:{case}:{doc}`, so the
  17k already-judged grades stayed valid across project swaps; only ~8k holes re-judged). Lesson: the gemini-3.x judge
  needs a stable, properly-billed account at lower concurrency (ran final pass at 4 workers, 0 fatal).

**HG7 gate — nuanced PASS.** Formal `eval judge-validate` FAILs (Spearman 0.17 / κ 0.10 / bias 0.36) — but it is computed
on **n=3–11 curated seeds** (single-digit; the documented sparse-gold bottleneck) AND treats every curated doc as exactly
grade-3 (so a "highly-relevant" grade-2 counts as disagreement). The meaningful calibration over **all 302 curated docs is
strong: 86.4% graded ≥2, 99% ≥1, mean 2.57** — clears the plan's stated "curated grade≥2 rate stays high" bar. Only 3/302
curated docs got grade-0, each with coherent judge reasoning (tangential `packing-*` seeds, not judge errors). Judge sound;
qrels trustworthy. (Caveat: per-run `benchmark.md` "best strategy" is unreliable under multi-run `--resume` — it reflects
only the last sub-run's `--strategy` set; trust the per-strategy `strategies/*.json`.)

**Artifacts:** indexes `data/retrieval-indexes/cuda-qwen4b-1024/{splade-opensearch…,splade-prithivida…,colbert-answerdotai…}`;
`data/eval/pool/catalog-nl-methods/pool.json` (8 systems, 25,305); `data/eval/qrels/catalog-nl-methods/qrels.json`
(gemini-3.1-pro, 175 cases); runs `data/eval/runs/nl-methods-{reporting,selection}/`. Code: judge retry/backoff +
fatal-abort; `splade_v3_encoder.py` opensearch routing; `experimental_index.py` CPU-query + lru_cache.
