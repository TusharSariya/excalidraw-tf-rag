---
name: profile-bench
description: Build an index for a named embed profile in graph-layout-rag or rag-literature-rag and benchmark it against qrels (nDCG@10/MRR/recall, dense+bm25+hybrid). Use whenever the user wants to try/compare/evaluate an embed profile, chunking variant, or retrieval strategy in either tool — phrases like "benchmark this profile", "try cuda-qwen4b-1024", "is section-v1 better than the baseline", "build and eval X", or "how does <profile> do on catalog/pdf-deep-read". Also trigger if the user names a profile that doesn't exist yet and wants to see numbers for it.
---

# profile-bench

Benchmarking an embed profile in this repo is always the same three-step
loop: rebuild the index for that profile, run the eval harness against a
qrels set, read the report. This skill is that loop, done right — the
biggest way to get it wrong is skipping the rebuild after an ingest option
changes, or benchmarking against qrels for the wrong track.

## Before you start

1. **Confirm the profile exists.** List built-in + tool-local profiles:
   ```bash
   cd tools/<graph-layout-rag|rag-literature-rag>
   uv run <tool> embed profiles
   ```
   Profiles live in `rag-common/src/rag_common/embed_profiles.toml` (shared
   across tools) merged with the tool's own `embed_profiles.toml`. If the
   user names something not in that list, tell them and ask whether they
   want a new profile added to the tool-local TOML first (that's a separate,
   code-authoring task — this skill only benchmarks existing profiles).

2. **This repo's embed backend is local-only** — no OpenAI/Gemini/MLX
   provider. Every profile runs real inference on this machine (CUDA/MPS/CPU
   depending on host and the profile's `quant`), so a rebuild is not free:
   graph-layout-rag's full corpus is ~4,500 docs and can take a while.
   Before running `--rebuild` on a full corpus, tell the user roughly what's
   about to happen (profile, corpus size if known, that it's a real compute
   job) and confirm — unless they've clearly already asked for exactly this
   (e.g. "just run it") or it's an explicit smoke test on a tiny corpus.

3. **Pick the qrels track(s).** Both tools have `catalog` and
   `pdf-deep-read`; graph-layout-rag additionally has NL variants
   (`catalog-nl`, `catalog-nl-methods`, `robust`, `robust-nl`) — list what
   exists with `ls data/eval/qrels/`. If the user didn't say which track,
   default to `catalog` and mention you're doing so.

## Run it

```bash
cd tools/<graph-layout-rag|rag-literature-rag>

# 1. Build the index for this profile (required whenever the profile/model
#    changes — an existing index for a different profile does NOT get reused).
RAG_EMBED_PROFILE=<profile> uv run <tool> ingest --force --rebuild -v

# 2. Benchmark against a qrels set, across all three retrieval strategies.
uv run <tool> eval benchmark --embed-profile <profile> \
  --qrels data/eval/qrels/<track>/qrels.json \
  --strategy dense --strategy bm25 --strategy hybrid --report -v
```

`--report` writes a markdown report next to the JSON output (check the
command's own output for the exact path — it's timestamped). Read that
report rather than re-deriving numbers from the raw JSON yourself.

If benchmarking more than one track, repeat step 2 per track — you do not
need to rebuild the index again for the same profile.

## Report the results

Summarize per strategy (dense / bm25 / hybrid): nDCG@10, MRR, recall@10.
Then, if you can find a comparable number, compare against the tool's
current production profile (`cuda-qwen0.6b-1024` in both tools as of this
writing — confirm from the tool's README rather than assuming it hasn't
changed) on the same track. Recent campaign write-ups often already have
this baseline number in `docs/quality-campaign-*.md` or
`data/eval/campaign/findings.md` — check there before re-running the
baseline yourself.

## What this skill does not do

- It does not change which profile is "production" or edit any config —
  report the numbers, let the user (or a separate, explicit task) decide
  whether to promote a profile.
- It does not add new profiles to `embed_profiles.toml` — that's a code
  change, not a benchmark run.
- It does not run `eval judge` / `eval pool` (regenerating qrels) — this
  skill benchmarks against *existing* qrels only. If qrels don't exist for
  a track/corpus the user cares about, say so rather than silently
  generating new ones (that's a separate, much more expensive judged-eval
  workflow).
