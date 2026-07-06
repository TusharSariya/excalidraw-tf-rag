#!/usr/bin/env bash
# Phase 2 glue: expose a v2-built index (keyed by corpus) as a v1-eval index
# (keyed by profile slug) so `eval benchmark --embed-profile <slug>` can read it.
#
# For each (corpus, profile) it builds  <EVAL_ROOT>/<slug>/  containing:
#   lancedb          -> symlink to pipeline-v2/src/data/indexes/<corpus>/lancedb
#   bm25             -> symlink to pipeline-v2/src/data/indexes/<corpus>/bm25
#   ingest_state.json-> synthesized embed-config so v1's match/HG1 guards pass
# The v2 index dirs are root-owned (pod-written) but world-readable, so eval
# reads them fine; we never write INTO them (no sudo) — the slug dir is ours.
#
# Then benchmark with:
#   GRAPH_RAG_INDEXES_DIR=<EVAL_ROOT> RAG_GPU_GATEWAY_URL=<gw> \
#     uv run graph-layout-rag eval benchmark --embed-profile cuda-qwen4b-1024 ...
set -euo pipefail
REPO=/home/tushar/excalidraw-tf-rag
V2_IDX="$REPO/pipeline-v2/src/data/indexes"
EVAL_ROOT="${EVAL_ROOT:-$REPO/pipeline-v2/data/eval-index-root}"

mkdir -p "$EVAL_ROOT"

# args: corpus slug embed_model embed_dims embed_profile embed_quant resolved_backend
make_slug() {
  local corpus="$1" slug="$2" model="$3" dims="$4" profile="$5" quant="$6" rbackend="$7"
  local src="$V2_IDX/$corpus" dst="$EVAL_ROOT/$slug"
  if [ ! -d "$src/lancedb" ]; then
    echo "SKIP $slug: $src/lancedb missing (build not done?)"; return 0
  fi
  rm -rf "$dst"; mkdir -p "$dst"
  ln -sfn "$src/lancedb" "$dst/lancedb"
  ln -sfn "$src/bm25" "$dst/bm25"
  python3 - "$dst/ingest_state.json" "$model" "$dims" "$profile" "$quant" "$rbackend" <<'PY'
import json, sys
path, model, dims, profile, quant, rbackend = sys.argv[1:7]
state = {
    "embed_backend": "local",
    "embed_model": model,
    "embed_dims": int(dims),
    "embed_profile": profile,
    "embed_quant": quant,
    "resolved_embed_backend": (None if rbackend in ("", "none", "None") else rbackend),
    "pdf_backend": "docling",
    "last_indexed_at": None,
}
json.dump(state, open(path, "w"), indent=2)
print("wrote", path)
PY
  echo "GLUE $slug -> $src  (lancedb+bm25 symlinked, ingest_state synthesized)"
}

# GLR: 4B index -> slug cuda-qwen4b-1024
make_slug glr cuda-qwen4b-1024 "Qwen/Qwen3-Embedding-4B" 1024 cuda-qwen4b-1024 4bit cuda-bnb-4bit
# RLR: 0.6B+contextual -> slug cuda-qwen0.6b-contextual-v1 (resolved_backend None as v1 recorded)
make_slug rlr cuda-qwen0.6b-contextual-v1 "Qwen/Qwen3-Embedding-0.6B" 1024 cuda-qwen0.6b-contextual-v1 4bit none

echo "EVAL_ROOT=$EVAL_ROOT"
ls -la "$EVAL_ROOT"
