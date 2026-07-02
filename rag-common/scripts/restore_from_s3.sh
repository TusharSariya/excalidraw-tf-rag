#!/usr/bin/env bash
# Restore RAG data from S3. Driven entirely by what's in the bucket (NOT by local
# files, and NOT by DynamoDB) so it rebuilds a machine whose data/ dir is empty.
#
# Derived artifacts are run-stamped (<tool>/<cat>/<name>/<run-id>/archive.tar.zst);
# restore picks the LATEST run per artifact by default, or the latest run at-or-
# before --as-of. Each tar object carries its data/-relative path in metadata, so a
# wiped machine lands files in the right place. Corpus is synced as-is.
#
#   ./restore_from_s3.sh rag-literature-rag                      # latest run of everything
#   ./restore_from_s3.sh rag-literature-rag mlx-qwen4b           # only artifacts matching a substring
#   ./restore_from_s3.sh graph-layout-rag corpus       # just the corpus
#   ./restore_from_s3.sh rag-literature-rag --as-of 20260630T0400Z   # latest run at/before a time
#   ./restore_from_s3.sh rag-literature-rag --dry-run            # list, touch nothing
#   ./restore_from_s3.sh rag-literature-rag mlx-qwen4b --force   # rm target dir before extract
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
export LC_ALL=C   # byte-wise string compare so run-id ordering is deterministic

DRY_RUN=0; FORCE=0; TOOL=""; FILTERS=(); ASOF=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --force) FORCE=1; shift ;;
    --as-of) ASOF="$2"; shift 2 ;;
    -h|--help) grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) die "unknown flag: $1" ;;
    *) if [[ -z "$TOOL" ]]; then TOOL="$1"; else FILTERS+=("$1"); fi; shift ;;
  esac
done
[[ -n "$TOOL" ]] || die "usage: restore_from_s3.sh <tool> [filter ...] [--as-of <run-id>] [--dry-run] [--force]"

BUCKET="$(resolve_bucket)"
DATA_DIR="$("$PY" "$SCRIPT_DIR/_manifest.py" --manifest "$MANIFEST" --data-root "$TOOLS_DIR" --datadirs --tool "$TOOL" | cut -f2)"
[[ -n "$DATA_DIR" ]] || die "tool '$TOOL' not in manifest"
log "bucket: s3://$BUCKET   tool: $TOOL   → $DATA_DIR   as-of: ${ASOF:-latest}   dry-run: $DRY_RUN"

# A key/artifact passes if no filters given, or it contains any filter substring.
match() { [[ ${#FILTERS[@]} -eq 0 ]] && return 0; local k="$1" f; for f in "${FILTERS[@]}"; do [[ "$k" == *"$f"* ]] && return 0; done; return 1; }
# Run-ids look like 20260630T034531Z-host. Guard against pre-run-stamped/legacy keys.
is_runid() { [[ "$1" =~ ^[0-9]{8}T[0-9]{6}Z ]]; }

n=0

# --- tarballs: pick the latest (or as-of) run per artifact -------------------
declare -A best_run best_key
while read -r key; do
  [[ -n "$key" && "$key" != "None" ]] || continue
  rest="${key#"$TOOL"/}"; noarch="${rest%/archive.tar.zst}"
  runid="${noarch##*/}"; artifact="${noarch%/*}"
  is_runid "$runid" || continue                       # skip legacy non-run-stamped keys
  match "$artifact" || continue
  [[ -n "$ASOF" && "$runid" > "$ASOF" ]] && continue  # skip runs after the cutoff
  if [[ -z "${best_run[$artifact]:-}" || "$runid" > "${best_run[$artifact]}" ]]; then
    best_run[$artifact]="$runid"; best_key[$artifact]="$key"
  fi
done < <(aws s3api list-objects-v2 --bucket "$BUCKET" --prefix "$TOOL/" \
           --query "Contents[?ends_with(Key, 'archive.tar.zst')].Key" --output text 2>/dev/null | tr '\t' '\n')

for artifact in "${!best_key[@]}"; do
  key="${best_key[$artifact]}"
  rel="$(aws s3api head-object --bucket "$BUCKET" --key "$key" --query 'Metadata.localrel' --output text 2>/dev/null)"
  [[ -n "$rel" && "$rel" != "None" ]] || rel="$artifact"
  target="$DATA_DIR/$rel"; parent="$(dirname "$target")"
  if [[ $DRY_RUN -eq 1 ]]; then printf '  untar [%s] → %s\n' "${best_run[$artifact]}" "$target" >&2; n=$((n+1)); continue; fi
  log "untar [${best_run[$artifact]}] → $rel"
  mkdir -p "$parent"; [[ $FORCE -eq 1 ]] && rm -rf "$target"
  aws s3 cp "s3://$BUCKET/$key" - --only-show-errors | zstd -d --long=31 -q | "$TAR" -C "$parent" -xf -
  n=$((n+1))
done

# --- corpus: sync each corpus/<x>/ prefix back to data/<x>/ -----------------
while read -r prefix; do
  [[ -n "$prefix" && "$prefix" != "None" ]] || continue
  match "$prefix" || continue
  x="$(basename "$prefix")"
  if [[ $DRY_RUN -eq 1 ]]; then printf '  sync  → %s/%s/\n' "$DATA_DIR" "$x" >&2; n=$((n+1)); continue; fi
  log "sync  → $x/"
  aws s3 sync "s3://$BUCKET/$prefix" "$DATA_DIR/$x/" --only-show-errors
  n=$((n+1))
done < <(aws s3api list-objects-v2 --bucket "$BUCKET" --prefix "$TOOL/corpus/" --delimiter / \
           --query 'CommonPrefixes[].Prefix' --output text 2>/dev/null | tr '\t' '\n')

# --- meta: <tool>/meta/<run-id>/<file> → latest run per file ----------------
declare -A meta_run meta_key
while read -r key; do
  [[ -n "$key" && "$key" != "None" ]] || continue
  rest="${key#"$TOOL"/meta/}"; runid="${rest%%/*}"; file="${rest#*/}"
  is_runid "$runid" || continue
  match "$file" || continue
  [[ -n "$ASOF" && "$runid" > "$ASOF" ]] && continue
  if [[ -z "${meta_run[$file]:-}" || "$runid" > "${meta_run[$file]}" ]]; then
    meta_run[$file]="$runid"; meta_key[$file]="$key"
  fi
done < <(aws s3api list-objects-v2 --bucket "$BUCKET" --prefix "$TOOL/meta/" \
           --query 'Contents[].Key' --output text 2>/dev/null | tr '\t' '\n')

for file in "${!meta_key[@]}"; do
  key="${meta_key[$file]}"
  if [[ $DRY_RUN -eq 1 ]]; then printf '  copy  [%s] → %s/%s\n' "${meta_run[$file]}" "$DATA_DIR" "$file" >&2; n=$((n+1)); continue; fi
  log "copy  [${meta_run[$file]}] → $file"
  aws s3 cp "s3://$BUCKET/$key" "$DATA_DIR/$file" --only-show-errors
  n=$((n+1))
done

SUFFIX=""; [[ $DRY_RUN -eq 1 ]] && SUFFIX=" (dry-run)"
log "done — restored $n item(s)${SUFFIX}"
[[ $n -eq 0 ]] && log "note: nothing matched (check tool name / filters / that a backup exists)"
exit 0
