#!/usr/bin/env bash
# GATE-1 end-to-end driver (runs on the desktop GPU box, in tmux).
# Assumes gen has ALREADY produced data/training/gate1/queries.jsonl.
# Serializes every GPU phase; fails LOUD (set -e) on any error.
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
ROOT="$HOME/excalidraw-tf/tools/graph-layout-rag"
export GRAPH_RAG_TRAIN_DATA_DIR="$ROOT/data/training/gate1"
export GRAPH_RAG_TRAIN_NAMESPACE="gate1-v1:"
export GRAPH_RAG_TRAIN_CORPUS_PROFILE="cuda-qwen4b-1024"
export GRAPH_RAG_ENCODE_DEVICE="cuda"
LOG="/tmp/gate1_driver.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== GATE1 DRIVER START $(date) ==="
echo "queries: $(wc -l < "$GRAPH_RAG_TRAIN_DATA_DIR/queries.jsonl") "
nvidia-smi --query-gpu=memory.used --format=csv,noheader
df -h / | tail -1

# 1) MINE (eval env, dense Qwen3-4B). Runs from ROOT.
echo "=== [1/5] MINE $(date) ==="
cd "$ROOT"
uv run --no-sync python training/build_triples.py --phase mine \
    --embed-profile cuda-qwen4b-1024 --mine-k 50 --mine-pool 30
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# 2) DENOISE (eval env, Qwen3-Reranker-4B nf4). OOM-safe config.
echo "=== [2/5] DENOISE $(date) ==="
uv run --no-sync python training/build_triples.py --phase denoise \
    --reranker Qwen/Qwen3-Reranker-4B --max-length 384 --batch-size 16 \
    --denoise-margin 0.10 --keep-negs 7 --min-negs 2 --text-chars 1000 \
    --max-candidates 0 --neg-cap 12
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# 3) SPLIT + DISTRACTOR POOL (eval env). Train/heldout by query; fixed shared pool.
echo "=== [3/5] SPLIT+POOL $(date) ==="
uv run --no-sync python training/gate1_split_and_pool.py \
    --heldout-frac 0.13 --distractors 300 --text-chars 1000

# 4) TRAIN on the TRAIN split ONLY (training env). Merged checkpoint + served!=base.
echo "=== [4/5] TRAIN $(date) ==="
cd "$ROOT/training"
uv run --no-sync python train_splade.py --epochs 3 \
    --triples "$GRAPH_RAG_TRAIN_DATA_DIR/triples_train.jsonl" \
    --out "$GRAPH_RAG_TRAIN_DATA_DIR/checkpoints/splade-gd-v1" \
    --batch-size 16 --lora-r 16 --lora-alpha 32 --query-reg 5e-4 --doc-reg 1e-4
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# 5) RANK TEST: stock vs fine-tuned on held-out positives (+ train sanity).
echo "=== [5/5] RANK TEST $(date) ==="
uv run --no-sync python gate1_rank_test.py \
    --heldout "$GRAPH_RAG_TRAIN_DATA_DIR/triples_heldout.jsonl" \
    --train "$GRAPH_RAG_TRAIN_DATA_DIR/triples_train.jsonl" \
    --distractors "$GRAPH_RAG_TRAIN_DATA_DIR/distractor_pool.jsonl" \
    --finetuned "$GRAPH_RAG_TRAIN_DATA_DIR/checkpoints/splade-gd-v1" \
    --device cuda --batch-size 32 --max-hard 7 --train-sanity-n 40 \
    --out "$GRAPH_RAG_TRAIN_DATA_DIR/gate1_rank_result.json"

echo "=== GATE1 DRIVER DONE $(date) ==="
nvidia-smi --query-gpu=memory.used --format=csv,noheader
df -h / | tail -1
echo "GATE1_DRIVER_EXIT=0"
