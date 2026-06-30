#!/usr/bin/env bash
# Shared helpers for backup_to_s3.sh / restore_from_s3.sh.
# Sourced, never executed directly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAG_COMMON_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TOOLS_DIR="$(cd "$RAG_COMMON_DIR/.." && pwd)"   # the submodule root that holds every <tool>/data
MANIFEST="$RAG_COMMON_DIR/backup-manifest.toml"

export AWS_PROFILE="${AWS_PROFILE:-admin}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

PY="${RAG_PYTHON:-python3}"
TAR="$(command -v gtar || command -v tar)"
# aws-cli may live in ~/.local/bin (desktop, no-sudo install) and not be on a
# non-interactive/cron PATH — resolve it explicitly.
AWS_BIN="${RAG_AWS_BIN:-$(command -v aws || echo "$HOME/.local/bin/aws")}"
[[ -x "$AWS_BIN" ]] || die "aws cli not found (set RAG_AWS_BIN); looked at: $AWS_BIN"
aws() { "$AWS_BIN" "$@"; }

# Storage class per backup category.
STORAGE_TAR="STANDARD_IA"      # index/extract tarballs: warm, restored occasionally
STORAGE_CORPUS="GLACIER_IR"    # source PDFs: instant-retrieval, rarely re-downloaded
STORAGE_META="STANDARD"        # tiny sqlite/json: keep hot

# zstd compression for tarballs. Default is high-but-practical: -19 is the top of
# the normal range. The vector indexes and PDF corpus are near-incompressible, so
# the "ultra" levels (-20..-22) mostly burn CPU for ~nothing on the big artifacts.
# Override for maximum: RAG_ZSTD_LEVEL="--ultra -22 --long=27". --long needs the
# same window on decompress; restore passes --long=31 to cover any window.
# shellcheck disable=SC2206
ZSTD_ARGS=(${RAG_ZSTD_LEVEL:--19 --long=27})

# Object tag that marks immutable run-stamped artifacts for the GLACIER lifecycle
# rule (corpus and _catalog/ are intentionally left untagged so they stay put).
LIFECYCLE_TAG_KEY="lifecycle-class"
LIFECYCLE_TAG_VAL="run-stamped"

die() { echo "error: $*" >&2; exit 1; }
log() { echo ">> $*" >&2; }

# Resolve the destination bucket: explicit env wins, else read aws-account-infra's
# `tofu output`. Override the infra location with RAG_AWS_INFRA_DIR.
resolve_bucket() {
  if [[ -n "${RAG_BACKUP_BUCKET:-}" ]]; then
    printf '%s' "$RAG_BACKUP_BUCKET"; return 0
  fi
  local infra="${RAG_AWS_INFRA_DIR:-$HOME/Projects/aws-account-infra}"
  [[ -d "$infra/stacks/rag" ]] || die "set RAG_BACKUP_BUCKET, or RAG_AWS_INFRA_DIR to your aws-account-infra checkout"
  local name
  name="$(tofu -chdir="$infra/stacks/rag" output -raw backup_bucket_name 2>/dev/null)" \
    || die "could not read backup_bucket_name from $infra/stacks/rag (run 'tofu init' there, or set RAG_BACKUP_BUCKET)"
  [[ -n "$name" ]] || die "backup_bucket_name output was empty"
  printf '%s' "$name"
}

# Emit manifest jobs as TSV: <tool>\t<category>\t<method>\t<s3_subpath>\t<local_path>
# Args: any number of --tool filters (passed through).
manifest_jobs() {
  "$PY" "$SCRIPT_DIR/_manifest.py" --manifest "$MANIFEST" --data-root "$TOOLS_DIR" "$@" \
    || die "manifest expansion failed (need python>=3.11 with tomllib)"
}

# Resolve the DynamoDB catalog table: explicit env wins, else aws-account-infra's
# `tofu output`, else the well-known default name (the desktop has no tofu state).
resolve_catalog_table() {
  if [[ -n "${RAG_BACKUP_CATALOG_TABLE:-}" ]]; then
    printf '%s' "$RAG_BACKUP_CATALOG_TABLE"; return 0
  fi
  local infra="${RAG_AWS_INFRA_DIR:-$HOME/Projects/aws-account-infra}"
  if [[ -d "$infra/stacks/rag" ]]; then
    local name
    name="$(tofu -chdir="$infra/stacks/rag" output -raw backup_catalog_table_name 2>/dev/null)" || name=""
    [[ -n "$name" ]] && { printf '%s' "$name"; return 0; }
  fi
  printf 'rag-backup-catalog'
}

human_size() { du -sh "$1" 2>/dev/null | cut -f1; }
