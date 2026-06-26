#!/usr/bin/env bash
# query_remote.sh — run a natural-language query on the desktop GPU box over SSH.
#
# The validated 4B index (cuda-qwen4b-1024) lives ONLY on the desktop (CUDA
# bnb-4bit). Quantized local vectors are not portable to the Mac's MLX backend,
# so the NL/4B path runs entirely on the desktop and streams JSON/text back.
# See docs/qwen-ladder-campaign-audit-log.md (M10-M16).
#
# Fails LOUD (never silently falls back to the 0.6B keyword path):
#   - desktop unreachable                 → exit 3
#   - 4B index not the validated state    → exit 4  (parity: chunks/model/dims)
#
# Usage: query_remote.sh "<query text>" [extra `graph-layout-rag query` flags]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOOL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${TOOL_ROOT}"

# Capture caller-provided values BEFORE sourcing .env so explicit environment
# overrides wins over .env (otherwise `set -a; source .env` clobbers them).
_PRE_SSH="${RAG_GPU_SSH:-${GRAPH_RAG_GPU_SSH:-}}"
_PRE_ROOT="${RAG_GPU_REMOTE_ROOT:-${GRAPH_RAG_GPU_REMOTE_ROOT:-}}"
_PRE_PROFILE="${GRAPH_RAG_NL_PROFILE:-}"
_PRE_SW="${GRAPH_RAG_NL_SPARSE_WEIGHT:-}"
_PRE_CHUNKS="${GRAPH_RAG_NL_EXPECTED_CHUNKS:-}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Resolve with precedence: caller env > .env > default.
SSH_HOST="${_PRE_SSH:-${RAG_GPU_SSH:-${GRAPH_RAG_GPU_SSH:-desktop}}}"
REMOTE_ROOT="${_PRE_ROOT:-${RAG_GPU_REMOTE_ROOT:-${GRAPH_RAG_GPU_REMOTE_ROOT:-excalidraw-tf}}}"

# NL/4B config — single-sourced here (documented in .env.example).
PROFILE="${_PRE_PROFILE:-${GRAPH_RAG_NL_PROFILE:-cuda-qwen4b-1024}}"
SW="${_PRE_SW:-${GRAPH_RAG_NL_SPARSE_WEIGHT:-0.4}}"
EXPECTED_CHUNKS="${_PRE_CHUNKS:-${GRAPH_RAG_NL_EXPECTED_CHUNKS:-41083}}"

if [[ $# -lt 1 ]]; then
  echo "query_remote.sh: missing query text" >&2
  echo "usage: query_remote.sh \"<query text>\" [extra query flags]" >&2
  exit 2
fi

# --- Preflight 1: desktop reachable -----------------------------------------
if ! ssh -o BatchMode=yes -o ConnectTimeout=8 "${SSH_HOST}" true 2>/dev/null; then
  echo "ERROR: desktop '${SSH_HOST}' unreachable — NL search needs the GPU box" \
       "powered on and SSH reachable. (keyword search still works locally)" >&2
  exit 3
fi

# --- Preflight 2: parity check (HG2) — index is the validated 4B state -------
# Fetch the built-index metadata and assert chunk count + model + dims match the
# campaign-validated cuda-qwen4b-1024. Cheap (no model load); catches a stale,
# half-synced, or mid-experiment desktop before we run the expensive query.
# shellcheck disable=SC2029  # intentional: ${REMOTE_ROOT} expands locally; $HOME/$PATH stay remote
INDEX_JSON="$(ssh "${SSH_HOST}" \
  "cd ~/${REMOTE_ROOT}/tools/graph-layout-rag && export PATH=\"\$HOME/.local/bin:\$PATH\" && uv run graph-layout-rag embed indexes --json 2>/dev/null")" || {
  echo "ERROR: could not read remote index metadata on '${SSH_HOST}'." >&2
  exit 4
}

# NOTE: the JSON is passed via env (INDEX_JSON), NOT piped to stdin — a
# `python3 - <<HEREDOC` reads its program from stdin, so a pipe would be ignored.
PARITY="$(INDEX_JSON="${INDEX_JSON}" PROFILE="${PROFILE}" EXPECTED_CHUNKS="${EXPECTED_CHUNKS}" python3 - <<'PY'
import json, os, sys

profile = os.environ["PROFILE"]
expected = int(os.environ["EXPECTED_CHUNKS"])
try:
    rows = json.loads(os.environ["INDEX_JSON"])
except Exception as exc:  # noqa: BLE001
    print(f"FAIL could not parse index metadata: {exc}")
    sys.exit(0)

row = next((r for r in rows if r.get("profile") == profile), None)
if row is None:
    print(f"FAIL profile {profile!r} not built on desktop")
    sys.exit(0)

chunks = row.get("chunks")
model = row.get("embed_model") or ""
dims = row.get("embed_dims")
if chunks != expected:
    print(f"FAIL chunk count {chunks} != validated {expected}")
elif "Qwen3-Embedding-4B" not in model:
    print(f"FAIL embed_model {model!r} is not Qwen3-Embedding-4B")
elif dims != 1024:
    print(f"FAIL embed_dims {dims} != 1024")
else:
    print(f"OK chunks={chunks} model={model} dims={dims}")
PY
)"

if [[ "${PARITY}" != OK* ]]; then
  echo "ERROR: desktop '${PROFILE}' index is not the validated state — ${PARITY#FAIL }." >&2
  echo "       Re-sync / re-embed the desktop before querying (refusing to return" \
       "results from a non-validated index)." >&2
  exit 4
fi

# --- Dispatch ----------------------------------------------------------------
# SSH does NOT preserve argv boundaries — it joins all args into one string that
# the remote shell re-splits on whitespace, so a multi-word query ("how do I …")
# would shatter. We therefore `printf '%q'`-quote every positional (root, profile,
# weight, query text, extra flags) into a single shell-safe string; the remote
# bash parses it back into the exact original tokens. Injection-safe.
# The heredoc script is read by the inner `bash -s` from stdin; the args are on
# its command line. Pinned NL flags come first so a caller can still override
# them (e.g. append --json, --top, or --no-hybrid).
REMOTE_ARGS="$(printf '%q ' "${REMOTE_ROOT}" "${PROFILE}" "${SW}" "$@")"
# shellcheck disable=SC2029  # intentional: REMOTE_ARGS is pre-quoted locally
ssh "${SSH_HOST}" "bash -s -- ${REMOTE_ARGS}" <<'REMOTE'
set -euo pipefail
ROOT="$1"; PROFILE="$2"; SW="$3"; shift 3
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/$ROOT/tools/graph-layout-rag"
# shellcheck disable=SC1091
source ../rag-common/scripts/gpu_env.sh 2>/dev/null || true
export RAG_LOCAL_EMBED_DEVICE=cuda
uv run graph-layout-rag query --embed-profile "$PROFILE" --hybrid --sparse-weight "$SW" "$@"
REMOTE
