#!/usr/bin/env bash
set -euo pipefail
export RAG_GPU_TOOL="${RAG_GPU_TOOL:-tools/graph-layout-rag}"
exec "$(cd "$(dirname "$0")/../../rag-common/scripts" && pwd)/gpu_sync_from_remote.sh" "$@"
