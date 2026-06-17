#!/usr/bin/env bash
# Thin wrapper — delegates to rag-common shared GPU scripts.
set -euo pipefail
export RAG_GPU_TOOL="${RAG_GPU_TOOL:-tools/graph-layout-rag}"
exec "$(cd "$(dirname "$0")/../../rag-common/scripts" && pwd)/gpu_dense_reembed.sh" "$@"
