#!/usr/bin/env bash
# Push source index + code to Ubuntu GPU box for dense re-embed.
set -euo pipefail

RAG_COMMON_SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
MONOREPO_ROOT="$(cd "${RAG_COMMON_SCRIPTS}/../../.." && pwd)"

RAG_GPU_TOOL="${RAG_GPU_TOOL:-tools/graph-layout-rag}"
TOOL_ROOT="${MONOREPO_ROOT}/${RAG_GPU_TOOL}"
cd "${TOOL_ROOT}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

SSH_HOST="${RAG_GPU_SSH:-${GRAPH_RAG_GPU_SSH:-desktop}}"
REMOTE_ROOT="${RAG_GPU_REMOTE_ROOT:-${GRAPH_RAG_GPU_REMOTE_ROOT:-excalidraw-tf}}"
REMOTE="${SSH_HOST}:~/${REMOTE_ROOT}"
SOURCE_PROFILE="${RAG_GPU_SOURCE_PROFILE:-${GRAPH_RAG_SOURCE_PROFILE:-gemini-2-structure-v1}}"
RAG_COMMON="${MONOREPO_ROOT}/tools/rag-common"
TOOL_REMOTE_NAME="$(basename "${RAG_GPU_TOOL}")"

echo "Syncing ${RAG_GPU_TOOL} to ${SSH_HOST} (source=${SOURCE_PROFILE})"

RSYNC_EXCLUDES=(
  --exclude '.venv'
  --exclude '__pycache__'
  --exclude '.pytest_cache'
  --exclude 'data/raw'
  --exclude 'data/bibliography-scan-cache'
  --exclude 'data/*.log'
  --exclude 'data/harvest*.log'
  --exclude 'data/ingest*.log'
  --exclude 'data/*.sqlite'
  --exclude 'data/*.sqlite-*'
  --exclude 'data/harvest.db'
  --exclude 'data/harvest_checkpoint.json'
  --exclude 'data/trusted_venue_checkpoint.json'
  --exclude 'data/graph.sqlite'
)

rsync -avz --progress "${RSYNC_EXCLUDES[@]}" \
  "${TOOL_ROOT}/" "${REMOTE}/tools/${TOOL_REMOTE_NAME}/"

rsync -avz --progress \
  --exclude '.venv' --exclude '__pycache__' \
  "${RAG_COMMON}/" "${REMOTE}/tools/rag-common/"

rsync -avz --progress \
  "data/indexes/${SOURCE_PROFILE}/" \
  "${REMOTE}/tools/${TOOL_REMOTE_NAME}/data/indexes/${SOURCE_PROFILE}/" 2>/dev/null || \
rsync -avz --progress \
  "data/lancedb/" \
  "${REMOTE}/tools/${TOOL_REMOTE_NAME}/data/lancedb/" 2>/dev/null || true

rsync -avz --progress \
  "data/indexes/${SOURCE_PROFILE}/ingest_state.json" \
  "${REMOTE}/tools/${TOOL_REMOTE_NAME}/data/indexes/${SOURCE_PROFILE}/ingest_state.json" 2>/dev/null || \
rsync -avz --progress \
  "data/ingest_state.json" \
  "${REMOTE}/tools/${TOOL_REMOTE_NAME}/data/ingest_state.json" 2>/dev/null || true

rsync -avz --progress \
  "data/indexes/${SOURCE_PROFILE}/bm25/" \
  "${REMOTE}/tools/${TOOL_REMOTE_NAME}/data/indexes/${SOURCE_PROFILE}/bm25/" 2>/dev/null || \
rsync -avz --progress \
  "data/bm25/" \
  "${REMOTE}/tools/${TOOL_REMOTE_NAME}/data/bm25/" 2>/dev/null || true

# citations.sqlite carries the `papers_meta` enrichment table that the query path
# joins by doc_id; the blanket `data/*.sqlite` exclude above keeps it out of the
# code rsync, so transfer it explicitly. WAL-checkpoint first so the single main
# file is self-contained (WAL/-shm are intentionally not synced). Set
# RAG_SYNC_CITATIONS=0 to skip.
SYNC_CITATIONS="${RAG_SYNC_CITATIONS:-1}"
if [[ "${SYNC_CITATIONS}" != "0" && -f data/citations.sqlite ]]; then
  if command -v sqlite3 >/dev/null 2>&1; then
    sqlite3 data/citations.sqlite 'PRAGMA wal_checkpoint(TRUNCATE);' >/dev/null 2>&1 || true
  fi
  echo "Syncing citations.sqlite (papers_meta enrichment) to ${SSH_HOST}"
  rsync -avz --progress \
    data/citations.sqlite \
    "${REMOTE}/tools/${TOOL_REMOTE_NAME}/data/citations.sqlite"
fi

echo "Done."
