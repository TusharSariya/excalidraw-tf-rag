#!/usr/bin/env bash
# query_remote.sh — run a rag-literature-rag query on the desktop GPU box over SSH.
#
# The 4B index (cuda-qwen4b-1024 / cuda-qwen4b-contextual-v1) lives ONLY on the
# desktop (CUDA bnb-4bit). Quantized CUDA vectors are not portable to the Mac's
# MLX backend, so the 4B path embeds + searches entirely on the desktop and
# streams JSON/text back. Ported from graph-layout-rag/scripts/query_remote.sh.
#
# Fails LOUD (never silently falls back to a local index):
#   - desktop unreachable              → exit 3
#   - 4B index not the expected state  → exit 4  (parity: chunks/model/dims)
#
# Usage: query_remote.sh "<query text>" [extra `rag-literature-rag query` flags]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOOL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${TOOL_ROOT}"

# Capture caller env BEFORE sourcing .env so explicit overrides win.
_PRE_SSH="${RAG_GPU_SSH:-${RAG_LIT_GPU_SSH:-}}"
_PRE_ROOT="${RAG_GPU_REMOTE_ROOT:-${RAG_LIT_GPU_REMOTE_ROOT:-}}"
_PRE_PROFILE="${RAG_LIT_NL_PROFILE:-}"
_PRE_CHUNKS="${RAG_LIT_NL_EXPECTED_CHUNKS:-}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

SSH_HOST="${_PRE_SSH:-${RAG_GPU_SSH:-${RAG_LIT_GPU_SSH:-desktop}}}"
REMOTE_ROOT="${_PRE_ROOT:-${RAG_GPU_REMOTE_ROOT:-${RAG_LIT_GPU_REMOTE_ROOT:-excalidraw-tf}}}"
PROFILE="${_PRE_PROFILE:-${RAG_LIT_NL_PROFILE:-cuda-qwen4b-1024}}"
EXPECTED_CHUNKS="${_PRE_CHUNKS:-${RAG_LIT_NL_EXPECTED_CHUNKS:-21211}}"

if [[ $# -lt 1 ]]; then
  echo "query_remote.sh: missing query text" >&2
  echo "usage: query_remote.sh \"<query text>\" [extra query flags]" >&2
  exit 2
fi

# --- Preflight 1: desktop reachable -----------------------------------------
if ! ssh -o BatchMode=yes -o ConnectTimeout=8 "${SSH_HOST}" true 2>/dev/null; then
  echo "ERROR: desktop '${SSH_HOST}' unreachable — the 4B index needs the GPU box" \
       "powered on and SSH reachable. (Use 'yarn rag-lit:query-local' for the local index.)" >&2
  exit 3
fi

# --- Preflight 2: parity check — index is the expected 4B state --------------
# shellcheck disable=SC2029
INDEX_JSON="$(ssh "${SSH_HOST}" \
  "cd ~/${REMOTE_ROOT}/tools/rag-literature-rag && export PATH=\"\$HOME/.local/bin:\$PATH\" && uv run rag-literature-rag embed indexes --json 2>/dev/null")" || {
  echo "ERROR: could not read remote index metadata on '${SSH_HOST}'." >&2
  exit 4
}

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
if not chunks:
    print(f"FAIL profile {profile!r} has 0 chunks (build incomplete)")
elif expected > 0 and chunks != expected:
    print(f"FAIL chunk count {chunks} != expected {expected}")
elif "Qwen3-Embedding-4B" not in model:
    print(f"FAIL embed_model {model!r} is not Qwen3-Embedding-4B")
elif dims != 1024:
    print(f"FAIL embed_dims {dims} != 1024")
else:
    print(f"OK chunks={chunks} model={model} dims={dims}")
PY
)"

if [[ "${PARITY}" != OK* ]]; then
  echo "ERROR: desktop '${PROFILE}' index is not the expected state — ${PARITY#FAIL }." >&2
  echo "       Re-embed/finish the desktop build before querying (refusing to return" \
       "results from a non-validated index)." >&2
  exit 4
fi

# --- Dispatch ----------------------------------------------------------------
# printf '%q'-quote every positional so a multi-word query survives SSH's
# argv-join-and-resplit. Pinned flags first; caller can append --json/--top/etc.
REMOTE_ARGS="$(printf '%q ' "${REMOTE_ROOT}" "${PROFILE}" "$@")"
# shellcheck disable=SC2029
ssh "${SSH_HOST}" "bash -s -- ${REMOTE_ARGS}" <<'REMOTE'
set -euo pipefail
ROOT="$1"; PROFILE="$2"; shift 2
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/$ROOT/tools/rag-literature-rag"
# shellcheck disable=SC1091
source ../rag-common/scripts/gpu_env.sh 2>/dev/null || true
export RAG_LOCAL_EMBED_DEVICE=cuda
uv run rag-literature-rag query --embed-profile "$PROFILE" --hybrid "$@"
REMOTE
