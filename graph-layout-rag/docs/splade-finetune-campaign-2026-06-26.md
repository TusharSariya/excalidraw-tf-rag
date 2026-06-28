# SPLADE domain-finetune campaign — execution log (2026-06-26)

**Question:** Can a SPLADE model *fitted to this graph-drawing corpus* beat the in-domain
`cuda-qwen4b-1024` dense / weighted-hybrid baseline, on keyword (`catalog`) and NL (`catalog-nl`)?
ColBERT is gated behind a SPLADE keyword win.

Plan: `~/.claude/plans/reflective-floating-dragon.md`. Admiral = main session (shit-test mode);
heavy work delegated to subagents. Prior context: stock SPLADE/ColBERT already NULL on NL (M18,
commit `972cd1e8e`); 4 prior nulls on this BM25-dominant corpus.

## Environment (verified 2026-06-26)
- Tool dir: `tools/graph-layout-rag`; eval harness present (`splade_v3_encoder.py`,
  `experimental_index.py`, `strategies/`, `benchmark.py`, `judge.py`).
- qrels: `data/eval/qrels/catalog/qrels.json` (keyword), `data/eval/qrels/catalog-nl/qrels.json` (NL),
  plus `catalog-nl-methods/` (the M18 re-pooled NL set). Folds: `data/eval/folds.json` (Jun 24).
- Judge cache: `data/eval/judge_cache.json` (11 MB, model-keyed → only new holes cost $).
- GPU: `ssh desktop` RTX 3060 Ti **8 GB**, FREE at start (17 MiB used, no 4B/ollama resident).
- Judging only on Mac (Vertex ADC). GPU jobs serialized on desktop.

## Gate ledger

| Gate | What | Cost | Status | Verdict |
|------|------|------|--------|---------|
| GATE -1 | Baseline-headroom / lexical-vs-semantic miss analysis | $0 | ✅ PASS (qualified) | best-chunk lexical 0.568 (kw) / 0.573 (nl); majority-lexical, not dominant; ~50% of beyond-pool misses chunk-recoverable |
| GATE 0 | Zero-shot released SPLADE + teacher (mxbai) sanity | $0 | ✅ DONE — NO teacher beats hybrid | all 4 rerankers lose (best Qwen3-4B −0.045; fulltext −0.066, no flip); 3rd reranking null → recipe pivot |
| **DECISION** | Teacher fork (admiral, autonomous per mandate) | — | ✅ Option A | pivot distillation→contrastive; Qwen3-4B as denoiser only; $0 to GATE 1; user may override (B/C/D) |
| Step 1 | Training env + label-free data (small batch), **contrastive** | $0 | ⏳ building | — |
| GATE 1 | Held-out-positive RANK test (non-circular) | $0 | ✅ PASS (qualified) | finetuned>stock held-out: MRR +0.059, r@1 +0.082, mean_rank −2.13; generalizes (not overfit); proxy=seed-chunk retrieval, not real eval |
| Step 2 | Wire splade-gd-v1 into harness | $0 | ✅ DONE | labels splade_gd/dense_splade_gd added; index built chunks=41083 (parity), served≠base (gd_nnz 462 vs base 191) |
| GATE 2 | Real-qrels keyword benchmark (on 1119-q ckpt) | $0 | ❌ NULL — fine-tune does NOT move real eval | splade_gd 0.536 < stock splade_os 0.570 (Δ −0.034); both far below dense 0.638 / hybrid 0.719; fusion dense_splade_gd 0.580 < dense_splade_os 0.608 |
| GATE 3 | De-biased keyword verdict (+ bootstrap CI, raw vs fused) | ~$3–12 | — | — |
| GATE 4 | De-biased NL verdict (only if keyword wins) | ~$ | — | — |
| GATE 5 | ColBERT (only if SPLADE wins keyword) | ~$ | — | — |

## Error watch (admiral monitors every subagent result for these)
OOM / CUDA OOM · killed processes (exit 137/143) · silent fallback to base model ·
empty rankings substituted for real ones · corrupt/empty JSON · chunk count ≠ 41,083 ·
judge cache poisoning (grade-0 frozen) · contamination (train query ≈ eval query).

## Timeline / findings
- **2026-06-26 — kickoff.** Plan approved. Environment verified (above). Launched Lane A (GATE -1)
  + Lane B (GATE 0) in parallel.
- **2026-06-26 — GATE -1 returned SOFT PASS.** Hybrid baseline reconstruction validated (mean
  nDCG@10 0.688 catalog ≈ documented 0.684–0.715). Not near-ceiling (median 0.706 catalog / 0.618 NL;
  only ~15% of queries ≥0.9). Miss split is the crux: vs the ~700-char pooled **excerpt** → semantic-
  dominant (lexical 0.24–0.33); vs **full concatenated doc text** → lexical-dominant (0.66–0.70).
  Strong recall signal: 45/88 (catalog) & 775/1153 (NL) missed-relevant docs were ranked *beyond pool
  depth-50* by hybrid entirely, yet contain query terms in full text.
  **Admiral shit-test:** full-doc concatenation is an *upper bound* (terms scattered across chunks
  count as covered though no single chunk has them); the excerpt is a *lower bound*. SPLADE retrieves
  at CHUNK granularity → the faithful proxy is **best-chunk coverage**. Commissioned that recompute
  (Lane A resumed, $0) before blessing PASS. Artifacts: `data/eval/headroom/`.
- **2026-06-26 — GATE -1 PASS (qualified).** Best-chunk (faithful SPLADE-grain) lexical fraction =
  **0.568 catalog / 0.573 NL**, between the excerpt lower bound (0.24–0.33) and full-doc upper bound
  (0.66–0.70) — did NOT collapse to excerpt. Median best-chunk coverage 0.50; of the never-surfaced
  (beyond-pool-50) misses, ~50% (24/45 kw, 388/775 nl) are chunk-recoverable (a recall hole, where
  term-expansion helps most). Honest read: majority-lexical but close to even (~57/43), cutoff-
  sensitive on a median-0.5 distribution → "real but modest target," not a slam dunk. GATE 1 remains
  the hard confirmation. Artifact: `data/eval/headroom/headroom_bestchunk_summary.json`.
- **2026-06-26 — GATE 0 returned: stock-SPLADE gap + TEACHER FAILED.**
  - *Task A:* M18 stock SPLADE on NL is ~0.09–0.10 below 4B-dense (best stock `splade_os` 0.634 vs
    0.727 leave-dense-out); keyword-track stock SPLADE was never run on the current baseline index (gap).
    A domain fine-tune must close a large gap just to reach baseline.
  - *Task B (teacher sanity):* `mxbai-rerank-large-v2` reranks the catalog pool at mean nDCG@10
    **0.650 vs same-pool weighted-hybrid 0.756 → −0.106**, loses 37/49 cases. Clean run (real GPU,
    one CUDA OOM caught + fixed on-GPU at max_length=512/bs=8, peak 4.69 GB, no CPU fallback, 49/49
    scored, GPU released to 17 MiB). **VERDICT: teacher too weak → distilling it would teach SPLADE a
    signal −0.106 below the baseline.** Consistent with this corpus's 4 prior nulls + 2 reranking nulls
    (general cross-encoders fail on a BM25-dominant corpus = outside-voice Risk #2).
  - *Admiral decision:* do NOT swap blindly. Resumed Lane B for a $0 teacher bake-off — does ANY
    strong off-shelf reranker beat the hybrid here? Also re-test the excerpt-truncation handicap
    (rerank fuller best-chunk text). Step 1 stays BLOCKED on a viable teacher.
    Artifacts: `scripts/teacher_sanity_rerank.py`, `data/eval/teacher_sanity/`.
- **2026-06-26 — GATE 0 teacher bake-off: NO off-shelf reranker beats hybrid (excerpt grain).**
  All on 49 catalog cases, same-pool weighted-hybrid = 0.756:
  | teacher | nDCG@10 | Δ | W/L/T |
  |---|---|---|---|
  | mxbai-rerank-large-v2 | 0.650 | −0.106 | 9/37/3 |
  | gte-reranker-modernbert-base | 0.675 | −0.081 | 11/36/2 |
  | Qwen3-Reranker-0.6B | 0.688 | −0.068 | 16/30/3 |
  | Qwen3-Reranker-4B (4-bit) | **0.710** | **−0.045** | 14/32/3 |
  Stronger models narrow but never cross the gap → structural: general-semantic rerankers can't beat
  the lexical hybrid on this BM25-dominant corpus (matches 4 prior nulls + 2 reranking nulls).
  Full-text (best-chunk) re-test of Qwen3-4B running to rule out the excerpt-truncation handicap.
  **Reframe:** the GATE -1 headroom is LEXICAL/recall; no available teacher injects that (all inject
  sub-baseline general relevance). BUT SPLADE's domain-lexical learning comes from FLOPS-regularized
  in-domain (query→source-chunk) contrastive training, NOT from a cross-encoder teacher. Qwen3-4B
  (0.71) is a competent false-negative *denoiser* though a poor *score-teacher*. So GATE 0 reshapes
  the training objective (distillation → contrastive) rather than necessarily killing the campaign.
  Escalating the fork to the user once the full-text number lands. Artifacts:
  `data/eval/teacher_sanity/bakeoff/`.
- **2026-06-26 — GATE 0 COMPLETE: no off-shelf teacher beats hybrid (3rd reranking null).**
  Final: excerpt best Qwen3-4B 0.710/−0.045 (W/L/T 14/32/3); full-text (best-chunk, ≤4 chunks/doc,
  max-pool) 0.690/−0.066 — marginally WORSE, **truncation hypothesis refuted**. Strength gradient
  (4B>0.6B>modernbert>mxbai) is real → conclusion = the reranking-as-teacher premise itself fails on
  this corpus, not the model choice. Clean run (2 CUDA OOMs both on-GPU-recovered, desktop disk-full +
  HF-xet-hang fixed, GPU released to 17 MiB). `jina-reranker-v3` deliberately skipped (bracketed by
  Qwen3 0.6B/4B; wouldn't change verdict).
- **2026-06-26 — DECISION (admiral, autonomous per "navigate through thick and thin" mandate):
  Option A — pivot training objective from cross-encoder distillation to in-domain CONTRASTIVE SPLADE.**
  Rationale: GATE 0 doesn't just fail the teacher, it reinforces the thesis — the GATE -1 headroom is
  lexical/recall, no semantic teacher can inject that, but SPLADE learns domain term-expansion from
  in-domain (synthetic-query→source-chunk) positives + BM25/dense hard negs + FLOPS sparsity reg,
  with NO cross-encoder teacher needed. Qwen3-4B (0.71, competent ranker) repurposed as a
  false-negative DENOISER (a role that doesn't require beating hybrid). Stays $0 until GATE 1 (the
  non-circular kill-gate) → bounded downside. Halt (D) considered but premature: GATE -1's recall hole
  (50% of never-surfaced misses chunk-recoverable) is exactly a learned-sparse retriever's wheelhouse;
  one $0-gated attempt is what the gate structure exists to permit before a 5th null. **User may
  override → (B) distill-from-hybrid-scores (tie-ceiling), (C) distill-from-judge ($), (D) halt.**
  Loss impl: `SpladeLoss(model, loss=SparseMultipleNegativesRankingLoss(model), query_reg=λ_q,
  doc_reg=λ_d)` — non-distillation SPLADE, sentence-transformers 5.x.
- **WATCH:** desktop disk 95% full (12 GB free) — training checkpoints + index builds need headroom.
- **2026-06-26 — Step 1 building (alive).** Training module written: `training/{gen_train_queries,
  build_triples,train_splade}.py + pyproject.toml + README + tests + _api_smoke/_base_probe}`. Agent
  proceeded past API-validation into data gen → strong signal the contrastive API path (SparseEncoder
  + SpladeLoss + SparseMultipleNegativesRankingLoss + LoRA merge) validated (confirm in report).
  Generating queries via LOCAL Ollama on desktop ($0, target 5000, concurrency 6) — correctly avoided
  cloud LLM. Disk trending 12→9.2 GB (one-time env+base install); watching.
- **2026-06-26 — Step 1 API VALIDATION: PASS (contrastive path confirmed).** Isolated training env on
  desktop (`training/pyproject.toml`, `uv sync`): torch `2.5.1+cu124`, sentence-transformers `5.5.1`,
  transformers `5.x`, peft `0.19.1` — coexists with eval env, does NOT touch eval `uv.lock`.
  `_api_smoke.py` RESULT ok=true: `SparseEncoder`/`SpladeLoss`/`SparseMultipleNegativesRankingLoss`
  all present; LoRA attaches (294,912 params on DistilBERT `q_lin/k_lin/v_lin/out_lin`);
  `merge_and_unload` → save → reload via `SparseEncoder(<path>)` works; **served≠base** (Δ=0.0044).
- **2026-06-26 — Step 1 BASE-SELECTION (decisive, validated on-GPU).** Probed all candidate SPLADE
  bases under the eval ABI (`_base_probe.py`). **Only `opensearch-project/opensearch-neural-sparse-
  encoding-doc-v3-distill` loads cleanly** (safetensors, plain DistilBERT-MLM, no `trust_remote_code`;
  harness routes `opensearch-*`; doc nnz≈154, asymmetric `Router` w/ inference-free
  `SparseStaticEmbedding` query branch). **Blocked bases (do NOT retry without changing eval torch
  pin):** `naver/splade-v3` = GATED; `naver/splade-v3-distilbert`, `prithivida/Splade_PP_en_v1`,
  `naver/efficient-splade-*` = transformers `torch.load≥2.6` security gate on legacy `.bin` (eval pins
  torch<2.6); `opensearch-...-doc-v3-gte` = custom-GTE `IndexError`/device-assert under transformers 5.x.
  Two training gotchas captured in README + `train_splade.py`: (1) Router needs
  `router_mapping={anchor:query, positive/negative:document}`; (2) `MLMTransformer.auto_model` doesn't
  round-trip the `PeftModel` (hold the `get_peft_model` ref locally for `merge_and_unload`).
- **2026-06-26 — Step 1 routing + tests.** `splade_v3_encoder.py` extended to route a local SPLADE
  checkpoint dir (path-component `splade`/`training` + `modules.json`) — eval-side suite 23/23 green.
  Training-env units 5/5 (dedup lexical+cosine leak drop + assert-fires; sibling exclusion; denoise
  drop). **Tiny E2E PASS:** 20 chunks → triples → 1-step contrastive LoRA → merge → serve via
  `SparseEncoder` → non-empty (nnz=141) + served≠base (Δ=9.5e-5).
- **2026-06-26 — Step 1 gen-tuning gotcha.** gemma3-QAT emits ~450-540 HIDDEN "thinking" tokens before
  the visible query; ANY finite `num_predict` (even 512) truncates to an EMPTY response
  (`done_reason=length`). Fix = leave `num_predict` UNSET (→`done_reason=stop`), no `stop` sequence.
  Cost: ~1.5-1.6 q/s sustained (0.68 q/s single-thread) — slow but $0/local. Target trimmed 5000→3500
  (plan's ≥3k floor) for a feasible first batch.
- **2026-06-26 — Step 1 DATA: switched gen to EXTRACTIVE (plan-sanctioned), full pipeline run.**
  The local LLM (gemma3-QAT) emits ~500 hidden tokens/query → ~0.3-0.7 q/s uncapped = too slow for a
  real batch. Switched `gen_train_queries.py` default to `--method extractive` (title + section + top
  content terms; instant, $0, NO GPU — runs on the Mac). **17,910 raw → 17,833 queries kept in ~16s**
  (15.5k q/s gen + 15s dedup); dedup dropped **66 lexical (Jaccard>0.45) + 11 cosine (>0.85)**, ASSERT-0
  survivors confirmed (max Jaccard 0.444, max cosine 0.849 — under both gates). `data/training/
  queries.jsonl` (gitignored). **Mining** (`build_triples --phase mine`, eval env w/ tantivy BM25 +
  Qwen3-4B dense, GRAPH_RAG_ENCODE_DEVICE=cuda): 3000-query subset → **2980 candidate rows**, avg ~24
  negs/query (siblings excluded; **0 sibling/self violations on 2000 rows**), ~3-4 q/s, GPU→17 MiB after.
  **Denoise COMPLETE** (Qwen3-Reranker-4B nf4 4-bit, max_length 384/batch 16, 300-cand subset, DENOISE_EXIT=0,
  no OOM in 8 GB): **253 triples**, avg 6.31 negs (min 2 / max 7), **1336 false-negatives dropped** (within
  0.10 margin of the positive's reranker score) → `triples.jsonl` w/ inline text + `positive_reranker_score`,
  NO teacher margins (contrastive MNRL). Full-scale gen/mine/denoise + production train = GATE-1's job.
- **2026-06-26 — Step 1 COMPLETE & validated.** API path PASS; base `opensearch-...-doc-v3-distill`
  (only one loadable under torch<2.6 — splade-v3 gated, others `.bin`-security-gated); tests 5/5 units
  + 23/23 routing + tiny-E2E served≠base. Pipeline ran end-to-end on REAL data: 17,833 dedup'd queries →
  2980 mined candidates → 253 denoised triples (~28% rows had false-negs dropped). GPU→17 MiB, disk 9.2 GB
  stable. **STOP before GATE 1** (held-out-positive rank test + production train = next unit). Nothing committed.
- **2026-06-26 — ADMIRAL: GATE-1 query-realism decision.** The extractive Step-1 data is
  pipeline-validation ONLY. GATE 1 will run on REALISTIC queries (local gemma4 `--method llm`, $0,
  ~0.5 q/s) — a ~2-2.5k batch is enough for a first held-out-rank signal (don't need 15k). Rationale:
  GATE 1 on verbatim-extractive queries is uninformative either way (pass=false, fail=ambiguous) →
  defeats the cheap-valid-kill-gate purpose; realistic queries create the query↔doc vocab gap that
  forces the term-EXPANSION the headroom needs. **$-question deferred behind the $0 gate:** if GATE 1
  PASSES, production batch (30-60k @ 0.5 q/s = 16-33 h local) → surface a minor Gemini-flash spend
  (~$3-5, minutes) to user; if GATE 1 FAILS → clean 5th null, halt $0. Launching GATE-1 subagent:
  waits for GPU-free, samples ~20 gemma4 LLM queries + self-checks realism (STOP+escalate if poor),
  else full gen → re-mine → re-denoise → train SPLADE-v1 LoRA → **non-circular held-out-positive RANK
  test** (rank true positive vs fixed distractor pool, MRR/recall on positions, fine-tuned vs stock —
  NOT teacher-score agreement). NOTE: `build_triples.py` runs in EVAL env (tantivy+bitsandbytes).
- **2026-06-26 — GATE 1 Step A: query-realism gate PASS (gemma4 LLM queries acceptable).** Denoise
  (tmux `dz`) finished → 253 `train-v1:` triples, GPU free (17 MiB). Generated a 37-query realism
  sample via `gen_train_queries.py --method llm` (gemma4:e4b-it-qat, local Ollama, $0) into a SEPARATE
  namespace/dir (`gate1-v1:` / `data/training/gate1/`, via new `GRAPH_RAG_TRAIN_DATA_DIR` +
  `GRAPH_RAG_TRAIN_NAMESPACE` env overrides — extractive `train-v1:` artifacts untouched). 37 raw → 37
  kept (dedup dropped 0 vs the 168 eval queries; under both gates). Styles split 13 keyword / 12 problem
  / 12 question. **Verdict: GOOD ENOUGH to proceed** — queries are coherent on-domain graph-drawing
  phrases (rectangular drawing, force-directed, edge bundling, book embeddings, metro maps, planar
  morphing), NOT verbatim chunk spans, NOT DOI/truncated-token junk like the extractive batch; the
  problem/question paraphrases create the query↔doc vocab gap that forces term-EXPANSION. Junk is a
  bounded minority (~16%: 3/37 LaTeX-math fragments, 3/37 paper-meta "acknowledgement/institution"),
  not dominant → does NOT trip the POOR→STOP-and-escalate condition. Observed gen rate ~0.15 q/s @
  concurrency 6 (slower than the 0.5 budget — gemma4-QAT's ~500 hidden tokens + 8GB single-model
  saturation). **Proceeding to Step B: full ~2200-query batch @ concurrency 10 (tmux `g1full`).**
- **2026-06-26 — GATE 1 RUNNING (realistic queries, quality PASS).** gemma4 `--method llm` sample
  (37 q) verified GOOD by admiral: diverse keyword/problem/question styles, coherent domain phrases
  ('Force directed layout node movement edge curves'), NO DOI/truncation junk — far better than
  extractive; minority of question-style are paper-specific trivia (acceptable noise). Full gen running
  in tmux `g1full` (`--target 750`, ~0.15 q/s → ~1.5-2 h, $0, gemma4 GPU). Downstream `gate1_driver.sh`
  (mine→denoise→split→train SPLADE-v1 LoRA→non-circular held-out rank test) BUILT + unit-tested by the
  agent, synced to desktop. **Continuation owned by admiral:** the GATE-1 agent paused expecting a
  self-Monitor (unreliable for subagents) → I drive the chain on gen-completion via SendMessage resume.
  Batch 750 is thin (small first signal); if borderline-positive, scale before any flash-$ spend.
- **2026-06-26 08:36 — GATE-1 gen COMPLETE ($0).** 1119 realistic queries kept (1121 raw, 2 jaccard-
  dropped, 0 cosine; `cloud_api_used:false`; gemma4; elapsed 6597s @ 0.17 q/s). gemma4 unloaded, GPU
  free, disk 9.2 GB. **Self-Monitor `bmft8pogn` did NOT auto-start the chain** (confirmed unreliable, as
  anticipated) → admiral resumed agent a22a246c via SendMessage to run `gate1_driver.sh` (mine→denoise→
  split→train SPLADE-v1 LoRA→non-circular held-out-positive rank test). Denoise on 1119 q may be slow
  (~0.13 cand/s in Step 1) — budgeted. Verdict pending: PASS→flash-$ production-gen decision to user;
  WEAK/FAIL→clean 5th-null halt.
- **2026-06-26 08:48 — GATE-1 chain running (admiral owns finish).** `gate1_driver.sh` in tmux
  `g1chain`: mine DONE (1118 rows) → DENOISE running (Qwen3-Reranker-4B, 1118 rows, ~2-2.5h long pole)
  → split (heldout-frac 0.13, 300-distractor fixed pool) → train splade-gd-v1 (LoRA, served≠base) →
  `gate1_rank_test.py` → **verdict file `data/training/gate1/gate1_rank_result.json`** (stock vs
  fine-tuned MRR/recall on held-out-positive rank positions + train-sanity). Rank test is properly
  non-circular. STOPPED agent a22a246c (was idle-pausing + spamming Monitor-wait notifications; chain
  runs independently in tmux). Admiral polls for completion → reads result JSON → interprets
  PASS/WEAK/FAIL directly. Denoise full (no cap) = more data, $0, unattended; poll watches OOM/disk.
- **2026-06-26 09:23 — GATE-1 TRAIN phase OOM'd (recovered).** Denoise+split completed (triples_train
  7.7MB / triples_heldout / distractor_pool all on disk). Train OOM'd mid-backward: `--batch-size 16`
  for SPLADE contrastive (query+pos+7negs+in-batch) exceeded 8 GB by ~648 MiB. **Admiral fix:** resume
  from train ONLY (upstream cached) via `~/g1_resume.sh` in tmux `g1train` — `--batch-size 4` +
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, then rank test (batch 16). No gen/mine/denoise
  redo. Denoise final: 1118 rows → ~928 triples, ~2834 false-negs dropped. Verifying batch-4 survives
  first steps before trusting the run.
- **2026-06-26 09:55 — GATE-1 train RECOVERED (batch 4).** No re-OOM. TRAIN_EXIT=0, 59s, 812 train/91
  eval rows, loss↓, **served_not_base=true (Δ0.00114)** — LoRA changed weights, no silent base-serving.
  merged_nnz=305. Rank test running → verdict imminent.
- **2026-06-26 10:10 — GATE 1 PASS (qualified, generalizing).** Non-circular held-out rank test
  (135 held-out gemma4 queries, fixed shared pool = 1 pos + 7 hard + 300 distractors):
  | metric | stock | finetuned | Δ |
  |---|---|---|---|
  | MRR | 0.746 | 0.805 | +0.059 |
  | recall@1 | 0.652 | 0.733 | +0.082 |
  | recall@5 | 0.859 | 0.904 | +0.044 |
  | mean_rank | 7.72 | 5.59 | −2.13 |
  **Not overfit:** held-out lift ≈ train-sanity lift (+0.069 MRR / +0.075 r@1 on 40 train q). The
  contrastive domain fine-tune helps retrieval AND generalizes to unseen queries.
  **CAVEAT (why this is a weak proxy):** it only proves better retrieval of the *generation-seed chunk*
  for held-out *gemma4* queries — near-guaranteed if training worked; does NOT prove transfer to the
  REAL eval (human/curated queries, judged-relevant docs). → Run GATE 2 (real-qrels keyword benchmark,
  $0) on the CURRENT 1119-q checkpoint BEFORE any flash-$ scaling decision. Defers $ behind one more
  $0 gate; gives user a real-eval-informed scale/halt call. splade-gd-v1: merged_nnz=305, served≠base.
- **2026-06-26 — USER DECISION: scale via Gemini-2.5-flash.** On a GATE-2 pass, generate the
  production batch (~30-60k queries) with Gemini-2.5-flash (~$3-5, minutes, eval-distribution-matched
  — same generator as the eval synth set; ADC already set up) rather than local-gemma4-overnight.
  Admiral proceeds autonomously on GATE-2 pass: flash gen → re-mine → denoise → retrain splade-gd-v2 →
  rebuild index → GATE 3 (judged keyword verdict, ~$3-12, judge cache model-keyed so only new holes
  cost). Interpreting intent as "go for the real verdict via flash." Still gated on GATE 2 ($0) showing
  the fine-tune moves the real keyword eval; if GATE 2 = no signal → halt, no spend.
- **2026-06-26 — CORRECTION: production gen model = Gemini-3.5-Flash** (newer than 2.5; repo already
  wired — `transform_cache_gemini_gemini-3.5-flash.json`, same Vertex/ADC as judge). Higher quality;
  the "same-generator-as-eval-synth (2.5)" distribution-match point only touched the NL synth subset,
  and the primary KEYWORD track uses curated queries → 3.5 quality wins; dedup gate guards
  contamination. NOTE: `gen_train_queries.py` currently has extractive + ollama-llm paths only; the
  scaling step must add a cloud-gen path reusing gold_synth's gemini client (rag_common.gemini_embed
  → client.models.generate_content) with model gemini-3.5-flash.
- **2026-06-26 — Step 2 DONE ($0).** Strategy labels `splade_gd` + `dense_splade_gd` added to
  `strategies/__init__.py` (EXPERIMENTAL_STRATEGIES + registry, mirroring stock `splade_os`/
  `dense_splade_os`; model comes from the wired `--retrieval-index`). Synced the two edited eval files
  (strategies + splade_v3_encoder routing) to the desktop (older commit, kept its dirty tree). Copied
  the missing `data/eval/folds.json` to desktop. **Served≠base probe (the key silent-failure check):**
  on the same doc, `splade-gd-v1` doc-vector nnz=462 vs stock base (`opensearch-...-doc-v3-distill`)
  nnz=191, term-Jaccard 0.28, mean |Δ| 0.090 on shared terms, 319 gd-only expansion terms → the merged
  fine-tune is materially different (heavy domain term-expansion), NOT silently the base. **Index built**
  over the full `cuda-qwen4b-1024` corpus: `eval build-retrieval-index --kind splade --model
  data/training/gate1/checkpoints/splade-gd-v1 --encode-device cuda --batch-size 8`, **chunks=41083
  (PARITY ✅)**, qdrant points=41083 status=green, fingerprint 3656964c2bb92b12 (4B corpus match), 565s,
  no OOM (peak ~3.8 GB / 8 GB), GPU→17 MiB, disk 7.9 GB.
- **2026-06-26 — GATE 2 = NULL. The domain fine-tune does NOT move the real keyword eval.** Two
  benchmark runs, catalog/reporting fold (49 cases), existing catalog qrels, $0 (no judge):
  | strategy | nDCG@10 | MRR | R@5 | HR@5 |
  |---|---|---|---|---|
  | splade_gd (fine-tuned) | **0.536** | 0.793 | 0.184 | 0.918 |
  | splade_os (stock, same base) | **0.570** | 0.795 | 0.203 | 0.939 |
  | dense_splade_gd (fused) | 0.580 | 0.864 | 0.203 | 0.939 |
  | dense_splade_os (fused) | 0.608 | 0.881 | 0.221 | 0.980 |
  | dense | 0.638 | 0.874 | 0.227 | 0.980 |
  | hybrid | 0.719 | 0.969 | 0.280 | 1.000 |
  **Cleanest same-hole-bias comparison (splade_gd vs stock splade_os, identical base + identical pool
  bias): −0.034 — the fine-tune is WORSE than stock on the real eval.** Fusion mirrors it (−0.028).
  Both SPLADE arms sit far below dense (−0.10/−0.07) and hybrid (−0.18/−0.14). Routing/serving verified
  honest: splade_gd ≠ splade_os in the results (0.536 vs 0.570), so this is a real null, not a
  silent-base artifact. **Interpretation:** the GATE-1 +0.059 MRR lift was a pure seed-chunk-retrieval
  proxy that did NOT transfer; on curated human queries vs judged-relevant docs the contrastive fine-tune
  on synthetic gemma4 queries *overfit to the synthetic-query distribution* and degraded real-eval
  ranking below the released checkpoint — the 5th null this BM25-dominant corpus has produced against the
  in-domain hybrid baseline. STOP per plan (GATE 2 "can't approach dense/hybrid → STOP"); no flash-$
  scaling, no GATE 3, no judge spend. Runs: `data/eval/runs/gate2-splade-gd-runA` (gd index + dense +
  hybrid), `gate2-splade-gd-runB` (stock opensearch-distill index). Nothing committed.
- **2026-06-26 10:55 — GATE 2: NULL (5th null). Campaign verdict.** splade_gd (fine-tuned) 0.536 <
  splade_os (stock, same base) 0.570 < dense_splade_gd 0.580 < dense_splade_os 0.608 < dense 0.638 <
  **hybrid 0.719** (catalog/reporting, 49 cases, existing qrels, $0). The fine-tune is −0.034 BELOW its
  own stock base — GATE-1's +0.059 proxy did NOT transfer (overfit synthetic gemma4-query distribution).
  Rigorous: served≠base verified two ways (gd doc nnz 462 vs base 191, term-Jaccard 0.28, 319 gd-only
  expansion terms — real fine-tune, not silent base); index parity 41,083; fairest same-base compare.
  **STRUCTURAL:** the whole SPLADE family (stock 0.570, fused 0.608) ceilings BELOW pure dense (0.638)
  and far below hybrid (0.719) → a 0.11–0.18 gap no fine-tune closes; adding SPLADE to dense HURTS
  (dense_splade_gd 0.580 < dense 0.638). Per GATE-2 STOP rule → halted, NO flash/judge spend, nothing
  committed. Run dirs: data/eval/runs/gate2-splade-gd-{runA,runB} (desktop). GPU released, disk 7.9 GB.
- **2026-06-26 — USER: budget cap raised $0.50 → $50.** Intent = give the flash-query hypothesis a
  real full-scale shot + fund the definitive judged verdict (GATE 3). Admiral flagged the structural
  ceiling (SPLADE family 0.570/0.608 < dense 0.638 < hybrid 0.719; SPLADE hurts fusion) — $50 buys a
  judged confirmation, but $0 GATE 2 already points null. Deployment (≤$50): (1) let cheap confounds
  finish (~$0.50, running) — the 1500-q flash A/B = directional read on flash-vs-gemma4 + gentle-retrain
  picks best config; (2) IF flash shows real lift → full-scale flash-3.5 gen (~30-60k) → retrain →
  GATE 3 judged keyword verdict (re-pool w/ splade_gd, judge ~$3-12, bootstrap CI, raw-vs-fused);
  (3) IF flash A/B identically dead → report + recommend against burning $50 on a structurally-capped
  approach (user decides). Don't spend the $50 reflexively.
- **2026-06-26 11:21 — T1 (gentle-retrain) results, parsed (CORRECTION).** Config A (1ep/lr5e-5/r8),
  strategy AGGREGATES: splade_gd **0.566** (≈ stock 0.570, just under), dense_splade_gd 0.609 (≈ stock
  fusion 0.608), **hybrid_splade_gd (3-way BM25+dense+SPLADE) 0.646 < hybrid 0.719**. (An earlier
  sit-rep cited 0.586/0.785 — those were PER-QUERY values, misread; true aggregates above.) Verdict:
  gentle training recovers the over-aggressive degradation (gd-v1 0.536 → 0.566) but lands AT stock,
  NOT above; and the 3-way shows **fine-tuned SPLADE DILUTES the hybrid (0.646 < 0.719), not augments**
  — closes the augmentation loophole. Deep structural reason: hybrid already has BM25 (lexical); SPLADE
  is learned-LEXICAL → largely redundant with BM25 on this BM25-dominant corpus → adds no orthogonal
  signal, only dilutes. This TIGHTENS the 5th null. T2 (flash A/B) pending. Disk 6.6 GB (watch).

## CAMPAIGN VERDICT (2026-06-26 ~11:50): 5th NULL, robust.
Can a corpus-fitted SPLADE beat the in-domain hybrid baseline (catalog keyword)? **NO.**
- GATE-1 proxy (+0.059 MRR held-out, seed-chunk retrieval) did NOT transfer to real eval.
- GATE-2: fine-tuned splade_gd 0.536 < stock splade_os 0.570 < dense 0.638 < hybrid 0.719.
- T1 confound (gentle retrain): over-training was real (0.536→0.566) but ceiling = STOCK (0.566≈0.570),
  never above. dense_splade_gd 0.609 ≈ stock-fused 0.608 < dense 0.638.
- **3-way hybrid_splade_gd 0.646 < hybrid 0.719 (−0.073): fine-tuned SPLADE DILUTES the hybrid** —
  learned-lexical SPLADE is redundant with BM25 on this BM25-dominant corpus → no orthogonal signal.
  This is the structural root of all 5 nulls (dense, semantic, graph, reranking ×3, now learned-sparse).
- T2 confound (flash query-quality A/B + GATE 3 judged verdict): **BLOCKED** — GCP account
  suspended/deleted/billing-disabled (403 CONSUMER_SUSPENDED / BILLING_DISABLED / invalid_grant). Needs
  USER to restore a billed GCP project + ADC. Cloud gen code is built+correct (gen_train_queries.py
  --method cloud --gen-model gemini-3.5-flash) but cannot run. NOTE: even if unblocked, T1's structural
  finding (SPLADE dilutes hybrid) makes T2 very likely to re-null — query quality can't fix redundancy.
- Spend: **$0** (all gates $0; flash never ran; no judge spend). $50 budget untouched (account dead).
- All rigor checks passed: served≠base verified every checkpoint, chunk parity 41,083, no OOM (after
  batch16→batch4 fix), no silent-base. Run dirs data/eval/runs/{gate2-splade-gd-*,t1-A,t1-B}.
RECOMMENDATION: HALT + write up. The null is robust on $0 evidence; structural ceiling (SPLADE⊂BM25)
means the one blocked confound (flash quality) can't overturn it. Pursue only if user wants the
definitive JUDGED verdict AND restores GCP auth.

## REOPENED — scaled flash run (user: "want to try that", project restored)
- **2026-06-26 — GCP auth FIXED.** User supplied billed project `project-c420f152-23bf-43d8-bec`.
  Updated `.env` GOOGLE_CLOUD_PROJECT/QUOTA_PROJECT (backup `.env.bak.*`). **Mac ADC was valid all
  along** (account alfrednobel694201; the dead-account errors were the DESKTOP's gcloud). Verified $0:
  gemini-3.5-flash (gen) ✅ + gemini-3.1-pro-preview (judge) ✅ both at location **global** (404 at
  us-central1). Cloud (gen+judge) runs on Mac; GPU on desktop.
- **2026-06-26 — DEFINITIVE scaled run launched (agent a600669406710fb18, ≤$50).** Fixes all 3 failure
  axes at once: better queries (flash-3.5 vs gemma4) + ~20k (real adaptation, not 1119-perturbation) +
  gentle training (T1 lesson). Pipeline: flash-3.5 gen ~20k (Mac, global, dedup gate) → mine → denoise
  Qwen3-4B (BOTTLENECK ~8-9h, overnight; subset+log if needed) → train splade-gd-v2 (gentle, batch4+
  expandable_segments, served≠base) → index (41,083) → **GATE 3 judged**: re-pool catalog w/ splade_gd
  (leave-dense-out, removes hole-bias) → judge gemini-3.1-pro holes-only (model-keyed cache) → benchmark
  reporting+selection+leave-dense-out + bootstrap CI, raw+fused+3-way vs dense/hybrid/stock. Honest
  expectation: structural SPLADE⊂BM25 redundancy likely still caps below hybrid, but this is the
  DEFINITIVE judged answer + the one untested confound (query quality+scale). ETA overnight → ~tomorrow.
- **2026-06-26 12:30 — SPEED: denoiser 4B→0.6B (user asked re parallelization).** Bottleneck was
  denoise (~5.5h on single 8GB GPU). Swapped Qwen3-Reranker-4B→0.6B (bake-off 0.688 vs 0.710 — ample
  for coarse false-neg filtering; ~5x faster, <2GB) → denoise ~5.5h→~1h, optionally lift cap toward
  full 20k. Pulls run in ~4.5h → finishes tonight. NOT parallelizing judge concurrency (memory: high
  concurrency burned GCP accounts — protect restored project). Bigger options noted-but-deferred:
  Mac-MLX denoise sharding (~2x), gen↔mine overlap (~2h). Pipeline still GPU-serialized on desktop.
- **2026-06-26 12:57 — gen RATE-LIMITED (caught).** Flash gen stalled ~6700/20000: 369× HTTP 429 +
  29× 403 + 24× 503 on project-c420f152 (new project, low Vertex quota for gemini-3.5-flash);
  concurrency-24 hammered it → effective rate collapsed 2.7→<0.08 q/s (ETA ~46h). FIX: kill (buffered
  ~6700 lost, ~$1.20), restart `--concurrency 6 --target 10000` (lower concurrency = higher real
  throughput under quota; 10k = 9x the original 1119, ample). Agent to report sustained q/s + 429 rate.
  If concurrency 3-4 STILL throttles → user must raise quota (GCP console: Vertex AI→Quotas→
  gemini-3.5-flash req/min). Cost cap $50.
- **2026-06-26 13:05 — gen rate-limit FIXED.** Restarted at concurrency 8 → 1.07 q/s sustained, ~0
  429s, 100% keep (c6 too slow 0.67 q/s; c24 was the throttle). Target 10k. ETA ~2.6h. No quota
  escalation needed. Brief PID race from restart cleaned (macOS no setsid; single clean proc verified).
  Spend ~$1.5. Revised verdict ETA ~20:00-21:00 tonight.

---

## TAIL-COLLAPSE finding (scaled flash gen, 2026-06-26 ~16:40 NDT)

**State:** cloud gen (gemini-3.5-flash, project-c420f152, `global`, concurrency 8) reached **7800/10000 completed futures, 7687 kept** at 16:02, then the tail **collapsed**: 16:02→16:36 the completed-future counter advanced **<100 in 34 min (<0.05/s)** while generateContent POSTs keep flowing at ~1.0/s. The POSTs are **retries inside the ~2200 still-pending tail futures**, not new completions.

**Mechanism (read from `training/gen_train_queries.py`):**
- cloud path builds a **fixed `--target`-sized job list, processed once** (`chunks[:args.target]`, HARD CAP, no oversample) — so it is NOT a dedup-spin; dedup runs once at the end (`dedup_against_eval`, line 483).
- `_gen_one_cloud` retry is **bounded** `for attempt in range(4)` with `sleep(min(retry_after or 2**attempt, 30))` → each future terminates (≤~120s worst case), so the run **will** finish.
- The tail is in a heavy-retry regime under the project's soft quota ceiling → effective completion ~0.05/s → **~11h ETA** for the last ~2200 jobs, which mostly **retry-exhaust to None** (≈0 added kept queries).

**Data-safety constraint:** output is written **only at clean loop exit** (`QUERIES_PATH.open("w")`, line 485). The ~7687 kept queries are **in-memory buffer**; killing the process **loses all of them**. No incremental checkpoint exists.

**Decision:** do NOT kill (preserves the 7687 buffer; killing is the only action that guarantees the loss). Let the bounded loop finish and write at line 485, then proceed to `gate3_driver.sh`. 7687 unique flash queries already **exceeds** the campaign need (rule out the query-quality confound; >> the gemma4 baseline set). Verdict ETA slips **tonight → ~tomorrow AM** unless the user authorizes a hard-stop + restart with smaller `--target`/incremental checkpointing (faster, but loses this buffer and re-spends).

**TODO (next run):** add incremental append-write (flush kept records to disk as they arrive) + a `--target`-of-kept early-stop, so a collapsed tail can't hold the whole buffer hostage.

## CRASH + RECOVERY (2026-06-26 ~17:15 NDT)

**Crash:** the scaled gen (PID 19276) died at ~16:40 on an **unhandled `httpx.RemoteProtocolError: Server disconnected`** raised through the google-genai SDK's tenacity retry, propagating out of `fut.result()` before the line-485 write → **all 7687 buffered queries lost** (output was written only at clean exit; no incremental checkpoint). Root defect in `_gen_one_cloud`: `if _is_fatal(exc) or not _is_rate_limit(exc): raise` misclassified a transient transport error as fatal.

**Fixes applied to `training/gen_train_queries.py` (uncommitted, gitignored module):**
1. `_gen_one_cloud`: treat `httpx.TransportError` (server-disconnect/conn-reset/timeout) as RETRYABLE, not fatal.
2. Per-future isolation in the cloud loop: a lone straggler exception is logged+skipped, not fatal.
3. **Incremental write:** flush each kept record to `queries.partial.jsonl` as it arrives → no future crash/OOM can discard the whole buffer.
4. `--target-kept N` early-stop: cancel the dead-weight tail once N kept collected (kills the ~11h/~0-yield tail).
5. **Circuit breaker:** 50 consecutive failures OR any `GeminiFatalError`/`_is_fatal` → abort LOUDLY (a systemic failure must not "succeed" with 0 kept).

**Self-inflicted detour caught (good test of #2/#5):** first hardened relaunch hit `GeminiFatalError ×10000` instantly (0 kept) because my `nohup` launch **didn't source `.env`** → `GOOGLE_CLOUD_PROJECT/USE_VERTEXAI/RAG_LLM_LOCATION` unset (location fell back to us-central1, no project). The per-future skip had *masked* it as a clean 0-kept run — exactly the silent-failure risk; the circuit breaker (#5) + treating `GeminiFatalError` as fatal now makes this abort loudly. Endpoint verified fine via a direct call once `.env` was sourced (`CALL OK`).

**Relaunched** (PID 18242, `.env` sourced, location=global, `--target 10000 --target-kept 7500 --concurrency 8 --seed 4242`). Healthy: CLOUD gen producing, partial file growing, 0 fatals. ETA ~7500 kept in ~2.7h (~20:00 NDT) → chain → judged GATE-3 verdict ~tonight (~22:00–23:00).

## GEN COMPLETE (clean, via early-stop) — 2026-06-26 19:20 NDT

Hardened relaunch finished correctly: `reached --target-kept=7500 at 7587 completed; cancelling 2413 stragglers` → `wrote 7475 queries -> data/training/gate3/queries.jsonl`. Dedup vs 168 eval queries dropped 25 (19 jaccard / 6 cosine) — contamination gate clean. 0 fatals/skips. ~2.05h wall, ~$ within cap (approx_in 2.70M tok, approx_out 0.21M tok). The early-stop cancelled the dead 2413-job tail (the failure mode that cost the first run).

**HOLDING before the desktop GPU chain (`gate3_driver.sh`).** User flagged the lost 2h and left a keep/cut/stop decision open; not auto-launching the next major (GPU) phase without go-ahead. Inputs staged & ready: queries.jsonl (7475), gate3_driver.sh (mine→0.6B-denoise→gentle-train splade-gd-v2→index), GATE-3 judged flow + bootstrap CI on the Mac.

## agy (Antigravity CLI) as a JUDGE backend — VALIDATED (2026-06-26 ~22:20 NDT)

GCP billing is dead (account `alfrednobel694201` deleted, `geoffbepsen` deleted, `bennettjeffrey990` billing CLOSED), blocking the Vertex gemini-3.1-pro judge. Tested routing the judge through the local **Antigravity CLI `agy`** (authenticated, off the dead billing; exposes Gemini 3.5 Flash + 3.1 Pro).

- **Gotcha (cost me a wrong "agy is dead" call):** `agy -p/--print` **silently drops stdout under a non-TTY** (pipe/subprocess/redirect) — appears to hang till timeout (GH issue #76). Fix = run under a **pseudo-terminal**. `script -q /dev/null` is flaky in this sandbox (`tcgetattr: not supported on socket`); the robust fix is a **Python `pty.openpty()` wrapper** (`$CLAUDE_JOB_DIR/tmp/agy_call.py`). Then strip ANSI + `\r`.
- **Latency:** ~9–13 s/call (Pro Low) under PTY — NOT the multi-minute "hang" the non-TTY drop faked.
- **Consistency sample (n=16, stratified by gold grade), agy "Gemini 3.1 Pro (Low)" vs cached Vertex `gemini-3.1-pro-preview`:** EXACT 12/16 (75%), within±1 16/16 (100%), **binary rel(≥2) 15/16 (94%)** — at the UMBRELA same-judge reliability ceiling. agy-Pro is a faithful drop-in.
- **Implication:** the judged GATE-3 is viable again via agy (new splade_gd holes only). Caveats: it's a different judge key (no cache reuse) + ToS/account-flagging risk from automated volume (keep volume low; user accepts). Sequence unchanged: run $0 GATE-2 first; only judge via agy if GATE-2 closes the gap.

**GPU chain:** MINE/DENOISE/TRAIN done (TRAIN served_not_base=true, 6061 triples). BUILD INDEX in progress (qdrant upsert growing ~9 MB/s). GATE-2 queued on INDEX_EXIT.

## agy judge HARDENED for unattended volume (2026-06-26 ~23:00 NDT)

Built `training/agy_judge.py` (resilient judge backend) + tests (`$JOB/tmp/test_agy_judge*.py`). All pass:
- **PTY** transport (fixes GH#76 non-TTY stdout drop).
- **validate_model() pre-flight** — REAL FINDING: agy *silently falls back to its default model* on an unknown `--model` string (a bogus name still returned 8 valid grades). So a typo'd model would judge the whole pool with the WRONG model undetected → refuse to start unless the exact string is in `agy models` (spinner-frame stripped).
- **Backoff retry** on transient (empty/timeout/parsefail/ratelimit; ratelimit gets 4× wait, cap 90s).
- **Circuit breaker** — N consecutive failures → abort loudly (rate-limit wall / agy down). Tested: fires, cache=0, no poison.
- **Auth/account-death** patterns → `JudgeFatal` abort (NOT a grade-0 sweep). Tested.
- **No grade-0 poisoning** — transient-exhausted holes stay UNCACHED (grade None). Tested (T7: 5 empty → 0 cached).
- **Incremental ATOMIC cache** (tmp+fsync+os.replace) → crash-resumable. Tested (T2 resume: judged=0 skipped=8).

Ready for GATE-3 judging via agy on a separate model key (`agy-pro-low:...`) once GATE-2 says whether a judge is even needed.

## AUTONOMOUS CONTINUATION PLAN (user asleep, 2026-06-26 ~23:05 NDT)

User authorized full autonomy to the verdict; stop only on a hard unblockable. Decision tree:

1. **GATE-2 (running, sequential, tmux `gate2`, /tmp/gate2.log, cap 26GB):** 3 splade arms vs stale catalog qrels. Baselines: dense 0.638, hybrid 0.719.
2. **If all splade arms remain well below dense (≤~0.60ish, big gap):** 5th NULL confirmed on scaled flash. NO judge needed (a judge only adds relevant docs the stale qrels miss — can't rescue a model already far behind on shared-pool nDCG). Write verdict + update memory `[[graph-rag-splade-finetune-null]]`. DONE.
3. **If any splade arm APPROACHES/exceeds dense (≥~0.62) — the gap closes:** the stale-qrels hole-bias may be hiding real gains → run de-biased GATE-3 via the hardened **agy judge** (`training/agy_judge.py`, validated κ=0.923 vs Vertex):
   - re-pool catalog with splade_gd/dense_splade_gd/hybrid_splade_gd added (`eval pool` + `pool-merge`).
   - judge ONLY new holes via agy "Gemini 3.1 Pro (Low)", key `agy-pro-low:`, on the Mac (off dead GCP billing). Resilient runner handles rate-limit/crash/resume.
   - benchmark reporting + selection + leave-dense-out + bootstrap CI on splade_gd vs dense/hybrid.
   - VERDICT with CIs. Update log + memory.
4. **Hard blockers that warrant waking the user:** agy account flagged/auth-dead (JudgeFatal, no fallback judge), desktop unreachable/OOM-looping, or GATE-2 can't produce any splade number after retries. Everything else: navigate around.

Cost: GATE-2 $0 (local). GATE-3 judge via agy = off dead billing (no GCP $). Within all constraints.

## AI Pro limits — judge pacing (2026-06-26 ~23:10, user has Google AI Pro)
agy draws on the AI Pro Gemini quota: **4× free-tier, refresh every ~5h, weekly cap** (exact numbers private; gemini.google.com→Settings→Usage Limits). Judge plan:
- Judge ONLY new splade_gd holes (few hundred), sequential ~10s/call → gentle, fits a 5h window.
- **Rate-limit circuit-breaker = PAUSE, not a blocker:** cache saved atomically, schedule resume after ~5h refresh, do NOT wake user. Resilient runner is resumable (model-keyed atomic cache).
- **Only auth/account-DEATH (JudgeFatal) wakes the user** (no fallback judge once GCP billing is dead).

## SPEEDUP: switched GATE-2 to qdrant SERVER mode (2026-06-26 ~23:31 NDT)
Root cause of slow GATE-2: benchmark ran qdrant in pure-Python LOCAL mode (single-thread, 100% one core, ~30-45min/arm) because `GRAPH_RAG_QDRANT_URL` was unset. **A Rust qdrant server (v1.18.2) is already running on the desktop at 127.0.0.1:6333** (Docker up). Code already supports it (`experimental_index.py:_connect`, env-gated). Fix = set `GRAPH_RAG_QDRANT_URL` + rebuild index INTO the server (local-format index not server-reusable → re-encode).
- Killed slow local GATE-2 (no splade results yet, nothing lost; dense 0.638/hybrid 0.719 already in hand).
- Rebuilding splade-gd-v2 into server (tmux `idxsrv`, /tmp/idx_server.log, collection `splade-...gd-v2-...20260627T020101Z`). GPU 95%.
- NEXT (autonomous): when build done → run benchmark with `GRAPH_RAG_QDRANT_URL=http://127.0.0.1:6333 --strategy splade_gd/dense_splade_gd/hybrid_splade_gd --retrieval-index <new dir> --track catalog --qrels data/eval/qrels/catalog/qrels.json --fold reporting`. Server search is multithreaded + low-RAM (index in server) → fast, no mutex/RAM-cap issues. Then resume decision tree (null vs GATE-3-via-agy).
- Fallback if server path fails: re-run local mode (cap 26), verdict still lands overnight.

## ★ VERDICT — scaled-flash GATE-2 (2026-06-26 23:45 NDT): 5th NULL CONFIRMED

Server-backed benchmark (qdrant Rust, ~2min vs ~1.5h local), catalog reporting fold, stale qrels:
| strategy | nDCG@10 | MRR | HR@5 | latency | failures |
|---|---|---|---|---|---|
| splade_gd (scaled-flash gd-v2) | **0.515** | 0.745 | 0.918 | 97ms | 3 |
| dense_splade_gd (fused) | **0.572** | 0.866 | 0.918 | 1129ms | 4 |
| hybrid_splade_gd (3-way) | **0.590** | 0.861 | 0.939 | 457ms | 2 |
| dense (baseline) | 0.638 | 0.874 | 0.980 | 671ms | 1 |
| hybrid (baseline) | 0.719 | 0.969 | 1.000 | 369ms | 0 |

**Scaled flash (7,475 gemini-3.5-flash queries) did NOT move the needle — slightly WORSE than the original campaign** (gd-v2 splade_gd 0.515 < prior gd 0.536 < stock 0.570 < dense 0.638 < hybrid 0.719).

**Bias-immune kill shot:** 3-way `hybrid_splade_gd` 0.590 is −0.129 BELOW plain 2-way hybrid 0.719 on the SAME qrels + same dense/bm25 core → adding scaled-flash SPLADE *dilutes* the hybrid (same dilution as the original 3-way 0.646<0.719, now worse). Hole-bias is symmetric here, so a de-biased judge can't flip it.

**Decision:** per pre-registered tree (judge only if a splade arm ≥~0.62; best is 0.590) → NULL branch, NO judge spent. The flash query-quality confound (left untested when GCP died) is now EMPIRICALLY RESOLVED: scaled/better flash queries do not help. Structural root unchanged — SPLADE is learned-lexical, redundant with the BM25 already in the hybrid on this lexically-saturated corpus.

Minor caveat: splade arms had 2–4 query failures (vs dense 1 / hybrid 0); even crediting them back (~+0.04) leaves splade_gd ~0.555 ≪ dense, hybrid_splade_gd ~0.63 < hybrid. Verdict robust.

**Optional (user's call, NOT spent autonomously):** a de-biased GATE-3 via the validated+hardened agy judge (κ=0.923, off dead GCP billing) is ready if the user wants gold-standard confirmation — but it's structurally moot given the dilution. Cost: ~1h agy + small AI-Pro-quota/account-flagging risk.

CAMPAIGN COMPLETE. Total cloud spend: gen ~$1-3 flash (now on dead account); GATE-2/index $0 local. Within all constraints.
