#!/usr/bin/env bash
# GATE-3 end-to-end GPU driver (desktop RTX 3060 Ti 8 GB, in tmux).
# Assumes gen has ALREADY produced data/training/gate3/queries.jsonl (flash-3.5).
# Chain: MINE -> DENOISE(subset-capped) -> TRAIN(gentle, served!=base) -> BUILD INDEX(41,083).
# GATE-3 JUDGING runs on the Mac (cloud) AFTER this. Fails LOUD on real errors.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT="$HOME/excalidraw-tf/tools/graph-layout-rag"
D="$ROOT/data/training/gate3"
PROFILE="cuda-qwen4b-1024"
CKPT="$D/checkpoints/splade-gd-v2"
DENOISE_CAP="${DENOISE_CAP:-0}"   # 0 = denoise ALL (0.6B is fast enough for full 20k)

export GRAPH_RAG_TRAIN_DATA_DIR="$D"
export GRAPH_RAG_TRAIN_NAMESPACE="gate3-v1:"
export GRAPH_RAG_TRAIN_CORPUS_PROFILE="$PROFILE"
export GRAPH_RAG_ENCODE_DEVICE="cuda"
LOG="/tmp/gate3_driver.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== GATE3 DRIVER START $(date) ==="
echo "queries: $(wc -l < "$D/queries.jsonl")  denoise_cap=$DENOISE_CAP"
nvidia-smi --query-gpu=memory.used --format=csv,noheader; df -h / | tail -1

# 1) MINE (eval env, dense Qwen3-4B). BM25 U dense top-50, sibling-excluded.
echo "=== [1/4] MINE $(date) ==="
cd "$ROOT"
uv run --no-sync python training/build_triples.py --phase mine \
    --embed-profile "$PROFILE" --mine-k 50 --mine-pool 30
echo "MINE_EXIT=$?"
echo "candidates: $(wc -l < "$D/candidates.jsonl" 2>/dev/null || echo NA)"
nvidia-smi --query-gpu=memory.used --format=csv,noheader; df -h / | tail -1

# 2) DENOISE (Qwen3-Reranker-0.6B — coarse false-neg filter only; ~5x faster than
# 4B, <2GB; GATE-0 bake-off 0.688 vs 4B 0.710 — irrelevant to a binary keep/drop).
# 0.6B is fast enough to denoise ALL mined candidates (cap=0 = no cap), so the
# full 20k feeds training (more denoised triples = better train).
echo "=== [2/4] DENOISE (0.6B, cap=$DENOISE_CAP) $(date) ==="
# NOTE: denoise honors --max-candidates (= #queries scored), NOT --max-queries
# (that only affects mine). cap=0 = denoise everything.
uv run --no-sync python training/build_triples.py --phase denoise \
    --reranker Qwen/Qwen3-Reranker-0.6B --max-length 384 --batch-size 32 \
    --denoise-margin 0.10 --keep-negs 7 --min-negs 2 --text-chars 1000 \
    --max-candidates "$DENOISE_CAP" --neg-cap 12
echo "DENOISE_EXIT=$?"
echo "triples: $(wc -l < "$D/triples.jsonl" 2>/dev/null || echo NA)"
nvidia-smi --query-gpu=memory.used --format=csv,noheader; df -h / | tail -1

# 3) TRAIN (gentle, batch-4 + expandable_segments). Merged + served!=base assert.
echo "=== [3/4] TRAIN (gentle 2ep/lr5e-5/r8) $(date) ==="
rm -rf "$CKPT"
cd "$ROOT/training"
uv run --no-sync python train_splade.py --epochs 2 \
    --triples "$D/triples.jsonl" --out "$CKPT" \
    --batch-size 4 --lr 5e-5 --lora-r 8 --lora-alpha 16 \
    --query-reg 5e-4 --doc-reg 1e-4 --eval-frac 0.1
echo "TRAIN_EXIT=$?"
nvidia-smi --query-gpu=memory.used --format=csv,noheader; df -h / | tail -1

# 4) BUILD SPLADE INDEX for splade-gd-v2 (must be chunks==41,083).
echo "=== [4/4] BUILD INDEX $(date) ==="
cd "$ROOT"
uv run --no-sync graph-layout-rag eval build-retrieval-index \
    --base-profile "$PROFILE" --kind splade --model "$CKPT" \
    --encode-device cuda --batch-size 8 2>&1 | tee /tmp/gate3_index.log
echo "INDEX_EXIT=$?"
IDX=$(ls -dt "$ROOT"/data/retrieval-indexes/"$PROFILE"/splade-*splade-gd-v2-* 2>/dev/null | head -1)
echo "INDEX_DIR=$IDX"
nvidia-smi --query-gpu=memory.used --format=csv,noheader; df -h / | tail -1

echo "=== GATE3 DRIVER DONE $(date) ==="
echo "GATE3_DRIVER_EXIT=0"
