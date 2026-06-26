#!/usr/bin/env bash
# query_auto.sh — single friendly entry that auto-routes a query to the right
# retrieval regime (see docs/qwen-ladder-campaign-audit-log.md M10-M16):
#
#   keyword / LLM-issued  → local 0.6B index (Mac, sparse-heavy hybrid)
#   human / natural-lang   → desktop 4B index over SSH (dense-leaning, +0.07 nDCG)
#
# The chosen backend is printed to stderr so the routing is never hidden. A
# wrong guess only changes which backend runs (never correctness); override with
# `--mode keyword|nl` (or GRAPH_RAG_SEARCH_MODE).
#
# Usage: query_auto.sh [--mode keyword|nl|auto] "<query text>" [extra query flags]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOOL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="${GRAPH_RAG_SEARCH_MODE:-auto}"
if [[ "${1:-}" == "--mode" ]]; then
  if [[ $# -lt 2 ]]; then
    echo "query_auto.sh: --mode needs a value (keyword|nl|auto)" >&2
    exit 2
  fi
  MODE="$2"
  shift 2
fi

if [[ $# -lt 1 ]]; then
  echo "query_auto.sh: missing query text" >&2
  echo "usage: query_auto.sh [--mode keyword|nl|auto] \"<query text>\" [extra flags]" >&2
  exit 2
fi

QUERY="$1"

# Resolve auto → keyword|nl via the Python classifier (source of truth, tested).
if [[ "${MODE}" == "auto" ]]; then
  MODE="$(cd "${TOOL_ROOT}" && uv run graph-layout-rag route "${QUERY}" 2>/dev/null || echo "")"
fi

case "${MODE}" in
  keyword)
    echo "[query-auto] routing to local-0.6B (keyword)" >&2
    cd "${TOOL_ROOT}"
    exec uv run graph-layout-rag query "$@"
    ;;
  nl)
    echo "[query-auto] routing to desktop-4B (nl)" >&2
    exec bash "${SCRIPT_DIR}/query_remote.sh" "$@"
    ;;
  *)
    echo "query_auto.sh: could not determine routing mode (got '${MODE:-empty}')." \
         "Pass --mode keyword|nl explicitly." >&2
    exit 2
    ;;
esac
