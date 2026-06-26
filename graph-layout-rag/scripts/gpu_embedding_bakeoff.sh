#!/usr/bin/env bash
# Stage B bakeoff driver: re-embed 4B then 8B Qwen3 from the 0.6B source index.
#
# Prerequisites:
#   - Phase-0 (0.6B source index) must be fully built on the GPU box first.
#   - Pool-merge qrels (data/eval/qrels/catalog-nl/qrels.json) must exist
#     (run gen-gold + pool + judge) before running eval against these indexes.
#
# Usage:
#   RAG_GPU_SSH=desktop RAG_GPU_TOOL=tools/graph-layout-rag \
#     bash tools/graph-layout-rag/scripts/gpu_embedding_bakeoff.sh
#
# To make executable: chmod +x tools/graph-layout-rag/scripts/gpu_embedding_bakeoff.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MONOREPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TOOL_ROOT="${MONOREPO_ROOT}/tools/graph-layout-rag"

# ── Common env ────────────────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAG_LLM_BACKEND=gemini
export RAG_LOCAL_EMBED_DEVICE=cuda
export RAG_GPU_TOOL="${RAG_GPU_TOOL:-tools/graph-layout-rag}"

# Load .env if present (provides RAG_GPU_SSH, RAG_GPU_REMOTE_ROOT, etc.)
cd "${TOOL_ROOT}"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

SSH_HOST="${RAG_GPU_SSH:-${GRAPH_RAG_GPU_SSH:-desktop}}"
REMOTE_ROOT="${RAG_GPU_REMOTE_ROOT:-${GRAPH_RAG_GPU_REMOTE_ROOT:-excalidraw-tf}}"
SOURCE_PROFILE="cuda-qwen0.6b-1024"

# ── Phase-0 completion gate ───────────────────────────────────────────────────
echo "=== Phase-0 gate: verifying 0.6B source index on ${SSH_HOST} ==="

ssh "${SSH_HOST}" bash -s -- "${REMOTE_ROOT}" "${SOURCE_PROFILE}" <<'GATE'
set -euo pipefail
REMOTE_ROOT="$1"
SOURCE_PROFILE="$2"
TOOL_DIR="${HOME}/${REMOTE_ROOT}/tools/graph-layout-rag"
INDEX_DIR="${TOOL_DIR}/data/indexes/${SOURCE_PROFILE}"

# a) Check chunk count > 0 in LanceDB table
LANCE_DIR="${INDEX_DIR}/lancedb"
if [[ ! -d "${LANCE_DIR}" ]]; then
  echo "ERROR: LanceDB dir not found: ${LANCE_DIR}" >&2
  exit 1
fi
CHUNK_COUNT=$(python3 -c "
import lancedb, sys
try:
    db = lancedb.connect('${LANCE_DIR}')
    tables = db.table_names()
    if not tables:
        print(0)
        sys.exit(0)
    tbl = db.open_table(tables[0])
    print(len(tbl))
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1) || { echo "ERROR: lancedb probe failed: ${CHUNK_COUNT}" >&2; exit 1; }
if [[ "${CHUNK_COUNT}" -le 0 ]]; then
  echo "ERROR: LanceDB table is empty (count=${CHUNK_COUNT}). Run 0.6B ingest first." >&2
  exit 1
fi
echo "  chunks in LanceDB: ${CHUNK_COUNT}"

# b) Check BM25 index dir exists
BM25_DIR="${INDEX_DIR}/bm25"
if [[ ! -d "${BM25_DIR}" ]]; then
  echo "ERROR: BM25 index dir not found: ${BM25_DIR}. Run 0.6B ingest first." >&2
  exit 1
fi
echo "  BM25 dir OK: ${BM25_DIR}"

# c) Check ingest_state.json exists and has finalized: true
STATE_FILE="${INDEX_DIR}/ingest_state.json"
if [[ ! -f "${STATE_FILE}" ]]; then
  echo "ERROR: ingest_state.json not found: ${STATE_FILE}. Run 0.6B ingest first." >&2
  exit 1
fi
FINALIZED=$(python3 -c "
import json, sys
with open('${STATE_FILE}') as f:
    s = json.load(f)
print(str(s.get('finalized', False)).lower())
")
if [[ "${FINALIZED}" != "true" ]]; then
  echo "ERROR: ingest_state.json has finalized=${FINALIZED}. Complete 0.6B ingest first." >&2
  exit 1
fi
echo "  ingest_state.json: finalized=true"

echo "Phase-0 gate PASSED."
GATE

echo ""

# ── Shared reembed invoker ────────────────────────────────────────────────────
# Delegates to the existing gpu_dense_reembed.sh thin wrapper, which in turn
# calls rag-common/scripts/gpu_dense_reembed.sh.  We override env vars so the
# shared script picks up the right profiles and batch sizes.
REEMBED_SCRIPT="${SCRIPT_DIR}/gpu_dense_reembed.sh"

run_reembed() {
  local target_profile="$1"
  local batch_size="$2"
  local label="$3"

  echo "=== ${label}: nvidia-smi snapshot on ${SSH_HOST} ==="
  ssh "${SSH_HOST}" "nvidia-smi" || true
  echo ""

  echo "=== ${label}: starting reembed (source=${SOURCE_PROFILE} → target=${target_profile}, batch=${batch_size}) ==="
  RAG_GPU_SOURCE_PROFILE="${SOURCE_PROFILE}" \
  RAG_GPU_TARGET_PROFILE="${target_profile}" \
  RAG_CUDA_BATCH_SIZE="${batch_size}" \
    bash "${REEMBED_SCRIPT}"
}

# ── Stage B-1: 4B reembed ─────────────────────────────────────────────────────
run_reembed "cuda-qwen4b-1024" "4" "Stage B-1 (Qwen3-4B)"

echo ""
echo "=== Cooling down GPU for 30 s before 8B model ==="
sleep 30

# ── Stage B-2: 8B reembed ─────────────────────────────────────────────────────
run_reembed "cuda-qwen8b-1024" "1" "Stage B-2 (Qwen3-8B)"

echo ""
echo "=== Stage B complete ==="

# ── Post-run assertion: pool-merge qrels ─────────────────────────────────────
QRELS_PATH="${TOOL_ROOT}/data/eval/qrels/catalog-nl/qrels.json"
if [[ ! -f "${QRELS_PATH}" ]]; then
  echo ""
  echo "REMINDER: Pool-merge qrels not found at:"
  echo "  ${QRELS_PATH}"
  echo "Run gen-gold + pool + judge before evaluating the new indexes."
fi
