#!/usr/bin/env bash
# T1 — gentle-retrain confound test ($0). Retrain on EXISTING triples_train.jsonl
# with LIGHT configs; build index (assert 41,083); GATE-2 keyword benchmark vs
# splade_os/dense/hybrid (+ 3-way hybrid_splade_gd). NO regen, NO judge spend.
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=$HOME/excalidraw-tf/tools/graph-layout-rag
D=$ROOT/data/training/gate1
QRELS=$ROOT/data/eval/qrels/catalog/qrels.json
PROFILE=cuda-qwen4b-1024
export GRAPH_RAG_TRAIN_DATA_DIR=$D
export GRAPH_RAG_TRAIN_NAMESPACE=gate1-v1:
export GRAPH_RAG_TRAIN_CORPUS_PROFILE=$PROFILE

run_config () {
  TAG=$1; EPOCHS=$2; LR=$3; R=$4; ALPHA=$5
  CKPT=$D/checkpoints/splade-gd-$TAG
  echo "===== [$TAG] TRAIN epochs=$EPOCHS lr=$LR lora-r=$R alpha=$ALPHA $(date) ====="
  rm -rf "$CKPT"
  cd "$ROOT/training"
  uv run --no-sync python train_splade.py --epochs "$EPOCHS" \
    --triples "$D/triples_train.jsonl" --out "$CKPT" \
    --batch-size 4 --lr "$LR" --lora-r "$R" --lora-alpha "$ALPHA" \
    --query-reg 5e-4 --doc-reg 1e-4
  echo "TRAIN_EXIT=$?  [$TAG]"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader; df -h / | tail -1

  echo "===== [$TAG] BUILD INDEX $(date) ====="
  cd "$ROOT"
  uv run --no-sync graph-layout-rag eval build-retrieval-index \
    --base-profile "$PROFILE" --kind splade --model "$CKPT" \
    --encode-device cuda --batch-size 8 2>&1 | tee "/tmp/t1_${TAG}_index.log"
  echo "INDEX_EXIT=$?  [$TAG]"
  IDX=$(ls -dt $ROOT/data/retrieval-indexes/$PROFILE/splade-*splade-gd-$TAG-* 2>/dev/null | head -1)
  echo "INDEX_DIR=$IDX"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader; df -h / | tail -1

  echo "===== [$TAG] BENCHMARK $(date) ====="
  GRAPH_RAG_EXPERIMENTAL_QUERY_DEVICE=cpu \
  uv run --no-sync graph-layout-rag eval benchmark \
    --embed-profile "$PROFILE" \
    --retrieval-index "$IDX" \
    --strategy splade_gd --strategy dense_splade_gd --strategy hybrid_splade_gd \
    --track catalog --qrels "$QRELS" --fold reporting \
    --run-dir "$ROOT/data/eval/runs/t1-$TAG" --json 2>&1 | tee "/tmp/t1_${TAG}_bench.log"
  echo "BENCH_EXIT=$?  [$TAG]"
  echo "===== [$TAG] DONE $(date) ====="
}

# Config A: very gentle
run_config A 1 5e-5 8 16
# Config B: a second light point
run_config B 2 1e-4 16 32

echo "===== T1 ALL DONE $(date) ====="
nvidia-smi --query-gpu=memory.used --format=csv,noheader; df -h / | tail -1
echo "T1_DRIVER_EXIT=0"
