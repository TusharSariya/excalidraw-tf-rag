# SPLADE domain-adaptation training (`training/`)

Isolated, **$0**, contrastive LoRA fine-tune of a released SPLADE checkpoint on the
graph-drawing corpus (Step 1 of the SPLADE domain-adaptation campaign). This module
is **not** part of the eval `uv.lock`; it has its own `pyproject.toml`/env and only
*reads* the eval corpus + reuses pure eval helpers.

> **Objective pivot (vs the original plan):** GATE 0 proved no off-the-shelf
> cross-encoder reranker beats the lexical hybrid baseline on this corpus, so there
> is no teacher worth distilling. We train **contrastively** —
> `SparseMultipleNegativesRankingLoss` (in-batch + hard negatives) under
> `SpladeLoss` FLOPS regularization — with **no teacher / no MarginMSE / no KL /
> no teacher_scores**. Qwen3-Reranker-4B is repurposed only as a *false-negative
> denoiser* for mined negatives.

## Validated API findings (critical — read before changing the base)

Run on the desktop GPU (`ssh desktop`, RTX 3060 Ti 8 GB), training env torch
`2.5.1+cu124`, sentence-transformers `5.5.1`, transformers `5.x`, peft `0.19.1`:

- **Contrastive APIs all present:** `SparseEncoder`, `SpladeLoss`,
  `SparseMultipleNegativesRankingLoss`. LoRA/PEFT attaches and `merge_and_unload`
  → save → reload through `SparseEncoder(<path>)` works, **served ≠ base** confirmed.
- **The only released SPLADE base that loads under this ABI is**
  `opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill`
  (safetensors, plain DistilBERT-MLM, no `trust_remote_code`; the harness already
  routes `opensearch-*`). It is **asymmetric**: a `Router` with an inference-free
  `SparseStaticEmbedding` **query** branch and a trainable DistilBERT **document**
  branch. LoRA targets = `q_lin, k_lin, v_lin, out_lin`.
- **Bases that do NOT work here** (do not retry without changing the eval torch pin):
  - `naver/splade-v3` — **gated** (HF auth/license).
  - `naver/splade-v3-distilbert`, `prithivida/Splade_PP_en_v1`,
    `naver/efficient-splade-*` — **blocked by the transformers `torch.load >= 2.6`
    security gate** (legacy `.bin` checkpoints), and the eval env is pinned
    `torch < 2.6`.
  - `opensearch-...-doc-v3-gte` — custom GTE code `IndexError`s / device-side-asserts
    under transformers 5.x.
- **Router training requirement:** the trainer needs
  `router_mapping={"anchor":"query","positive":"document","negative":"document"}`.
- **LoRA/merge gotcha:** the `MLMTransformer.auto_model` attribute does not round-trip
  the `PeftModel` wrapper (reads back as the bare HF model), but the LoRA params are
  still registered in the module tree and ARE trained. Hold the `get_peft_model(...)`
  return value in a local variable and call `merge_and_unload()` on **that** — see
  `train_splade.py` / `_api_smoke.py`.

## Files

| File | Purpose |
|---|---|
| `pyproject.toml` | Isolated env (s-t 5.5.x, torch>=2.4,<2.6, transformers, peft, bitsandbytes, lancedb). |
| `gen_train_queries.py` | Training-query generation over LanceDB chunks (DEFAULT `--method extractive`: title+section+content-term queries, instant/$0/no-GPU; optional `--method llm` local Ollama, slow) + **hard dedup gate** vs the 175 eval queries (token-Jaccard > 0.45 OR cosine > 0.85 dropped; assert-0 survivors). |
| `build_triples.py` | BM25∪dense hard-neg mining (siblings excluded) + Qwen3-Reranker-4B false-negative denoise → `triples.jsonl` (inline text, ~7 negs, no teacher margins). |
| `train_splade.py` | Contrastive LoRA fine-tune → `merge_and_unload` → standalone checkpoint + served≠base assert. |
| `test_training.py` | Unit tests: dedup lexical+cosine leak drop + assert-fires; sibling exclusion; denoise drop. |
| `test_e2e_tiny.py` | `[→E2E]` 20-chunk smoke: triples → 1-step train → merge → serve → non-empty + served≠base. |
| `_api_smoke.py`, `_base_probe.py` | One-shot API-validation probes (kept for re-validation). |

## Run (on the desktop GPU box)

```bash
ssh desktop
export PATH="$HOME/.local/bin:$PATH"
cd ~/excalidraw-tf/tools/graph-layout-rag/training
uv sync                       # build the isolated env (~5.7 GB, torch+cu124)

# fast unit tests
uv run --no-sync python -m pytest test_training.py -q
# tiny end-to-end (downloads the SPLADE base once)
uv run --no-sync python -m pytest test_e2e_tiny.py -q -s

# 1) generate training queries -- DEFAULT = extractive (instant, $0, NO GPU; runs
#    on the Mac too). The local-LLM path (gemma4:e4b) works but is ~0.5 q/s (gemma3-QAT
#    spends ~500 hidden tokens/query) so it is too slow for a real batch -> extractive.
export GRAPH_RAG_TRAIN_CORPUS_PROFILE=cuda-qwen4b-1024   # or cuda-qwen0.6b-1024 locally
uv run --no-sync python gen_train_queries.py --method extractive --target 12000
#    (LLM variant: --method llm  + RAG_OLLAMA_HOST / RAG_TRAIN_GEN_MODEL=gemma4:e4b-it-qat)

# 2) mine + denoise triples. IMPORTANT: build_triples runs in the EVAL env (it needs
#    tantivy BM25 + bitsandbytes; it does NOT need the SPLADE training stack). Run it
#    from the graph-layout-rag ROOT, not training/. Dense embedder (mine) and the
#    Qwen3-Reranker-4B (denoise) are NEVER co-resident in 8 GB -- separate phases.
cd ~/excalidraw-tf/tools/graph-layout-rag
uv run --no-sync python training/build_triples.py --phase mine   --embed-profile cuda-qwen4b-1024 --max-queries 3000
uv run --no-sync python training/build_triples.py --phase denoise --reranker Qwen/Qwen3-Reranker-4B

# 3) contrastive LoRA train -> merged checkpoint (TRAINING env, from training/)
cd training
uv run --no-sync python train_splade.py --epochs 2 \
    --out ../data/training/checkpoints/splade-gd-v1
```

> **Env split:** `gen_train_queries.py` (extractive) and `build_triples.py` run in the
> **eval** env (corpus + retrieval + reranker). `train_splade.py` + the tiny E2E run in
> the **training** env (`s-t 5.5.x` + peft). The merged checkpoint then serves back in
> the eval env via `SparseEncoder(<path>)`.

Artifacts land in `data/training/` (gitignored). **STOP after Step 1** — GATE 1
(held-out-positive rank test) and the full production training are the next unit.

## Contamination firewall

`gen_train_queries.py` dedups every generated query against **all 175 eval query
strings** (49 curated `GOLD_CASES` ∪ 126 synthetic gold cases). Drop if token-Jaccard
> 0.45 OR MiniLM cosine > 0.85; a post-filter assertion at the canonical thresholds
(`ASSERT_JACCARD_MAX`/`ASSERT_COSINE_MAX`) is the safety net even if the filter gates
are loosened. The `train-v1:` namespace marks training-only data. gold_synth's
eval-facing dedup is **not** touched.
