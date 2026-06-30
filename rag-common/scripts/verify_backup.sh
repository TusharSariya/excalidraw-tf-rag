#!/usr/bin/env bash
# Verify the RAG backup is complete and current enough to trust a "nuke local and
# rebuild". For every artifact that exists locally, assert the latest backup is
# FRESH (its content signature matches what's on disk now) and its S3 object is
# really there. Exits non-zero if anything is STALE or MISSING — so it works as a
# pre-nuke gate.
#
#   ./verify_backup.sh                         # every tool present locally
#   ./verify_backup.sh --tool repo-rag         # one tool
#   ./verify_backup.sh --only mlx              # only artifacts matching a substring
#   ./verify_backup.sh --drill repo-rag/indexes/mlx-qwen4b
#                                              # round-trip one artifact S3→disk→checksum
#
# Uses the DynamoDB catalog for the freshness check; without it, falls back to a
# bucket-only existence check (a backup object exists for each artifact).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
export LC_ALL=C

TOOL_ARGS=(); ONLY=(); DRILL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tool) TOOL_ARGS+=(--tool "$2"); shift 2 ;;
    --only) ONLY+=("$2"); shift 2 ;;
    --drill) DRILL="$2"; shift 2 ;;
    -h|--help) grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown arg: $1" ;;
  esac
done

BUCKET="$(resolve_bucket)"
TABLE="$(resolve_catalog_table)"
HAVE_CATALOG=1
aws dynamodb describe-table --table-name "$TABLE" >/dev/null 2>&1 || HAVE_CATALOG=0

match_only() { [[ ${#ONLY[@]} -eq 0 ]] && return 0; local s="$1" f; for f in "${ONLY[@]}"; do [[ "$s" == *"$f"* ]] && return 0; done; return 1; }
is_runid() { [[ "$1" =~ ^[0-9]{8}T[0-9]{6}Z ]]; }

catalog_latest() {
  local artifact="$1"
  [[ $HAVE_CATALOG -eq 1 ]] || { printf '\t'; return 0; }
  aws dynamodb query --table-name "$TABLE" \
    --key-condition-expression 'artifact = :a' \
    --expression-attribute-values "{\":a\":{\"S\":\"$artifact\"}}" \
    --no-scan-index-forward --limit 1 \
    --projection-expression 'content_sig, s3_key' \
    --output json 2>/dev/null | "$PY" "$SCRIPT_DIR/_catalog.py" extract-latest 2>/dev/null || printf '\t'
}

# Latest run-stamped object key for an artifact, bucket-only (no catalog).
latest_object_key() {
  local tool="$1" subpath="$2" best="" run key
  while read -r key; do
    [[ -n "$key" && "$key" != "None" ]] || continue
    run="${key%/archive.tar.zst}"; run="${run##*/}"
    is_runid "$run" || continue
    [[ -z "$best" || "$run" > "${best%$'\t'*}" ]] && best="$run"$'\t'"$key"
  done < <(aws s3api list-objects-v2 --bucket "$BUCKET" --prefix "$tool/$subpath/" \
             --query "Contents[?ends_with(Key, 'archive.tar.zst')].Key" --output text 2>/dev/null | tr '\t' '\n')
  printf '%s' "${best#*$'\t'}"
}

object_exists() { aws s3api head-object --bucket "$BUCKET" --key "$1" >/dev/null 2>&1; }

# --- drill: prove one artifact round-trips S3 → disk → identical bytes -------
if [[ -n "$DRILL" ]]; then
  log "drill: $DRILL"
  tool="${DRILL%%/*}"; subpath="${DRILL#*/}"
  data_dir="$("$PY" "$SCRIPT_DIR/_manifest.py" --manifest "$MANIFEST" --data-root "$TOOLS_DIR" --datadirs --tool "$tool" | cut -f2)"
  key=""
  if [[ $HAVE_CATALOG -eq 1 ]]; then latest="$(catalog_latest "$DRILL")"; key="${latest#*$'\t'}"; fi
  [[ -n "$key" ]] || key="$(latest_object_key "$tool" "$subpath")"
  [[ -n "$key" ]] || die "drill: no backup object found for $DRILL"
  rel="$(aws s3api head-object --bucket "$BUCKET" --key "$key" --query 'Metadata.localrel' --output text 2>/dev/null)"
  [[ -n "$rel" && "$rel" != "None" ]] || rel="$subpath"
  scratch="$(mktemp -d -t ragdrill.XXXXXX)"; trap 'rm -rf "$scratch"' EXIT
  log "drill: restoring $key → scratch"
  aws s3 cp "s3://$BUCKET/$key" - --only-show-errors | zstd -d --long=31 -q | "$TAR" -C "$scratch" -xf -
  restored="$scratch/$(basename "$rel")"
  live="$data_dir/$rel"
  got="$("$PY" "$SCRIPT_DIR/_catalog.py" content-sig "$restored")"
  want="$("$PY" "$SCRIPT_DIR/_catalog.py" content-sig "$live")"
  if [[ "$got" == "$want" ]]; then
    log "drill PASS — restored bytes match live $rel (content-sig $got)"
    exit 0
  fi
  die "drill FAIL — restored $rel differs from live (got $got want $want)"
fi

# --- completeness/freshness sweep ------------------------------------------
pass=0; stale=0; missing=0
verify_one() {
  local artifact="$1" local_path="$2" subpath="$3" tool="$4"
  local local_sig; local_sig="$("$PY" "$SCRIPT_DIR/_catalog.py" content-sig "$local_path")"
  if [[ $HAVE_CATALOG -eq 1 ]]; then
    local latest sig key; latest="$(catalog_latest "$artifact")"; sig="${latest%%$'\t'*}"; key="${latest#*$'\t'}"
    if [[ -z "$sig" ]]; then printf '  MISSING  %s  (no catalog row)\n' "$artifact" >&2; missing=$((missing+1)); return; fi
    if [[ "$sig" != "$local_sig" ]]; then printf '  STALE    %s  (local changed since last backup)\n' "$artifact" >&2; stale=$((stale+1)); return; fi
    if ! object_exists "$key"; then printf '  MISSING  %s  (catalog row but object gone: %s)\n' "$artifact" "$key" >&2; missing=$((missing+1)); return; fi
    printf '  PASS     %s\n' "$artifact" >&2; pass=$((pass+1))
  else
    local key; key="$(latest_object_key "$tool" "$subpath")"
    if [[ -n "$key" ]] && object_exists "$key"; then printf '  PASS*    %s  (bucket-only, existence)\n' "$artifact" >&2; pass=$((pass+1))
    else printf '  MISSING  %s  (no backup object)\n' "$artifact" >&2; missing=$((missing+1)); fi
  fi
}

log "verify: bucket=s3://$BUCKET  catalog=$([[ $HAVE_CATALOG -eq 1 ]] && echo "$TABLE" || echo OFF)"
while IFS=$'\t' read -r tool category method s3_subpath local_path local_rel; do
  [[ -n "$tool" ]] || continue
  [[ "$method" == "sync" ]] && continue   # corpus: incremental sync, represented by its meta manifest
  artifact="$tool/$s3_subpath"
  match_only "$artifact" || continue
  verify_one "$artifact" "$local_path" "$s3_subpath" "$tool"
done < <(manifest_jobs "${TOOL_ARGS[@]}")

log "verify done — PASS=$pass STALE=$stale MISSING=$missing"
[[ $stale -eq 0 && $missing -eq 0 ]] || exit 1
exit 0
