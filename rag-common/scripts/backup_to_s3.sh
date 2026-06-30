#!/usr/bin/env bash
# Back up RAG data (indexes, corpus, docling extract, eval, metadata) to S3.
#
# Filesystem-driven: only what exists locally is backed up, so the SAME command is
# correct on the desktop (corpus + CUDA/gemini/jina indexes) and on the Mac (mlx-*
# builds). Derived artifacts are IMMUTABLE and run-stamped under a per-run id, and
# unchanged artifacts are deduped (a new catalog row points at the prior object
# instead of re-uploading). See backup-manifest.toml for the key layout.
#
#   ./backup_to_s3.sh                      # every tool, everything present locally
#   ./backup_to_s3.sh --tool repo-rag      # one tool
#   ./backup_to_s3.sh --only mlx           # only artifacts whose key matches a substring
#   ./backup_to_s3.sh --dry-run            # print the plan, touch nothing
#   ./backup_to_s3.sh --no-dedup           # re-upload even unchanged artifacts
#   ./backup_to_s3.sh --no-catalog         # skip DynamoDB + the run manifest (S3 only)
#
# Dual-machine note: the desktop is authoritative for corpus + cuda/gemini/jina
# builds — back those up there with a plain run. On the Mac, push ONLY the mlx-*
# builds it owns: `--only mlx`, run last so its mlx is authoritative.
#
# Bucket:  $RAG_BACKUP_BUCKET, else `tofu output` from aws-account-infra.
# Catalog: $RAG_BACKUP_CATALOG_TABLE, else `tofu output`, else rag-backup-catalog.
# Creds:   $AWS_PROFILE (desktop: rag-backup; Mac: admin).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

DRY_RUN=0
TOOL_ARGS=()
ONLY=()
CATALOG=1
DEDUP=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tool) TOOL_ARGS+=(--tool "$2"); shift 2 ;;
    --only) ONLY+=("$2"); shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --no-catalog) CATALOG=0; shift ;;
    --no-dedup) DEDUP=0; shift ;;
    -h|--help) grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown arg: $1" ;;
  esac
done

# A job passes if no --only given, or its artifact key contains any --only substring.
match_only() { [[ ${#ONLY[@]} -eq 0 ]] && return 0; local s="$1" f; for f in "${ONLY[@]}"; do [[ "$s" == *"$f"* ]] && return 0; done; return 1; }

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$(hostname -s | tr -cd 'a-zA-Z0-9-')"
CREATED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
HOST="$(hostname -s)"
TOOL_GIT_SHA="$(git -C "$TOOLS_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
BUCKET="$(resolve_bucket)"
TABLE=""
[[ $CATALOG -eq 1 ]] && TABLE="$(resolve_catalog_table)"

RUN_NDJSON="$(mktemp -t ragbak-run.XXXXXX)"
cleanup() { rm -f "$RUN_NDJSON" 2>/dev/null || true; }
trap cleanup EXIT

log "bucket: s3://$BUCKET   run: $RUN_ID   profile: $AWS_PROFILE   catalog: ${TABLE:-off}   dedup: $DEDUP   dry-run: $DRY_RUN"

n_tar=0 n_sync=0 n_copy=0 n_dedup=0 bytes=0

# --- catalog helpers -------------------------------------------------------
# Newest catalog row for an artifact, as "content_sig<TAB>s3_key" (empty on miss).
catalog_latest() {
  local artifact="$1"
  [[ $CATALOG -eq 1 && $DEDUP -eq 1 ]] || { printf '\t'; return 0; }
  aws dynamodb query --table-name "$TABLE" \
    --key-condition-expression 'artifact = :a' \
    --expression-attribute-values "{\":a\":{\"S\":\"$artifact\"}}" \
    --no-scan-index-forward --limit 1 \
    --projection-expression 'content_sig, s3_key' \
    --output json 2>/dev/null | "$PY" "$SCRIPT_DIR/_catalog.py" extract-latest 2>/dev/null || printf '\t'
}

# Record one artifact: append to the run manifest and best-effort PutItem. All
# fields arrive via CATREC_* env so bash never hand-assembles JSON.
record_artifact() {
  local rec
  rec="$("$PY" "$SCRIPT_DIR/_catalog.py" make-record)" || { log "warn: make-record failed ($CATREC_ARTIFACT)"; return 0; }
  printf '%s\n' "$rec" >> "$RUN_NDJSON"
  [[ $CATALOG -eq 1 ]] || return 0
  local item
  item="$(printf '%s' "$rec" | "$PY" "$SCRIPT_DIR/_catalog.py" to-item)" || { log "warn: to-item failed ($CATREC_ARTIFACT)"; return 0; }
  aws dynamodb put-item --table-name "$TABLE" --item "$item" >/dev/null 2>&1 \
    || log "warn: catalog put-item failed for $CATREC_ARTIFACT (S3 object + run manifest still authoritative)"
}

# Tag a run-stamped object so the GLACIER lifecycle rule applies and the object is
# self-describing. Best-effort: a missing tag only loses cost-tiering, not data.
tag_object() {
  local key="$1" tool="$2" category="$3" name="$4" sha="$5"
  aws s3api put-object-tagging --bucket "$BUCKET" --key "$key" --tagging \
    "TagSet=[{Key=$LIFECYCLE_TAG_KEY,Value=$LIFECYCLE_TAG_VAL},{Key=tool,Value=$tool},{Key=category,Value=$category},{Key=name,Value=$name},{Key=run_id,Value=$RUN_ID},{Key=sha256,Value=$sha}]" \
    >/dev/null 2>&1 || log "warn: put-object-tagging failed for $key (lifecycle tiering skipped)"
}

# Common record emit for a tar/copy artifact (uploaded or deduped).
emit_record() {
  local artifact="$1" tool="$2" category="$3" name="$4" s3_key="$5" sha="$6" size="$7" sig="$8" storage="$9" deduped="${10}" provenance="${11}" version="${12}"
  CATREC_ARTIFACT="$artifact" CATREC_RUN_ID="$RUN_ID" CATREC_TOOL="$tool" \
  CATREC_CATEGORY="$category" CATREC_NAME="$name" CATREC_S3_BUCKET="$BUCKET" \
  CATREC_S3_KEY="$s3_key" CATREC_S3_VERSION_ID="$version" CATREC_SHA256="$sha" \
  CATREC_SIZE_BYTES="$size" CATREC_CONTENT_SIG="$sig" CATREC_STORAGE_CLASS="$storage" \
  CATREC_DEDUPED="$deduped" CATREC_HOST="$HOST" CATREC_TOOL_GIT_SHA="$TOOL_GIT_SHA" \
  CATREC_CREATED_AT="$CREATED_AT" CATREC_PROVENANCE="$provenance" \
    record_artifact
}

# Pull a little build provenance out of an index dir (best-effort, may be empty).
read_provenance() {
  local dir="$1" f
  for f in ingest_status.json ingest_state.json; do
    if [[ -f "$dir/$f" ]]; then
      head -c 4000 "$dir/$f" | tr -d '\n' ; return 0
    fi
  done
}

# tar a directory → one run-stamped zstd object, deduped against the prior run.
do_tar() {
  local local_path="$1" tool="$2" category="$3" name="$4" subpath="$5" local_rel="$6"
  local artifact="$tool/$subpath" key="$tool/$subpath/$RUN_ID/archive.tar.zst"
  local parent base size sig latest prev_sig prev_key
  parent="$(dirname "$local_path")"; base="$(basename "$local_path")"
  size="$(du -sk "$local_path" | awk '{print $1*1024}')"
  sig="$("$PY" "$SCRIPT_DIR/_catalog.py" content-sig "$local_path")"

  if [[ $DRY_RUN -eq 1 ]]; then
    printf '  tar   %-11s %8s  → s3://%s/%s\n' "$STORAGE_TAR" "$(human_size "$local_path")" "$BUCKET" "$key" >&2
    n_tar=$((n_tar+1)); bytes=$((bytes + size)); return
  fi

  latest="$(catalog_latest "$artifact")"; prev_sig="${latest%%$'\t'*}"; prev_key="${latest#*$'\t'}"
  if [[ -n "$prev_sig" && "$prev_sig" == "$sig" && -n "$prev_key" ]]; then
    log "dedup $local_rel  (unchanged → $prev_key)"
    emit_record "$artifact" "$tool" "$category" "$name" "$prev_key" "" "0" "$sig" "$STORAGE_TAR" "true" "$(read_provenance "$local_path")" ""
    n_dedup=$((n_dedup+1)); return
  fi

  log "tar  $(human_size "$local_path")  $local_rel  → $RUN_ID"
  local shafile; shafile="$(mktemp -t ragbak-sha.XXXXXX)"
  "$TAR" -C "$parent" -cf - "$base" \
    | zstd "${ZSTD_ARGS[@]}" -T0 -q \
    | tee >(sha256sum | cut -d' ' -f1 > "$shafile") \
    | aws s3 cp - "s3://$BUCKET/$key" \
        --storage-class "$STORAGE_TAR" --expected-size "$size" \
        --metadata "localrel=$local_rel,kind=tar-zst,run_id=$RUN_ID" --only-show-errors
  local sha objsize version
  sha="$(cat "$shafile")"; rm -f "$shafile"
  objsize="$(aws s3api head-object --bucket "$BUCKET" --key "$key" --query 'ContentLength' --output text 2>/dev/null || echo 0)"
  version="$(aws s3api head-object --bucket "$BUCKET" --key "$key" --query 'VersionId' --output text 2>/dev/null || echo '')"
  [[ "$version" == "None" ]] && version=""
  tag_object "$key" "$tool" "$category" "$name" "$sha"
  emit_record "$artifact" "$tool" "$category" "$name" "$key" "$sha" "$objsize" "$sig" "$STORAGE_TAR" "false" "$(read_provenance "$local_path")" "$version"
  n_tar=$((n_tar+1)); bytes=$((bytes + size))
}

# sync a corpus tree as-is (one object per file; incremental; NOT run-stamped).
do_sync() {
  local local_path="$1" tool="$2" subpath="$3"
  local prefix="$tool/$subpath"
  if [[ $DRY_RUN -eq 1 ]]; then
    printf '  sync  %-11s %8s  → s3://%s/%s/\n' "$STORAGE_CORPUS" "$(human_size "$local_path")" "$BUCKET" "$prefix" >&2
    n_sync=$((n_sync+1)); return
  fi
  log "sync $(human_size "$local_path")  $prefix"
  aws s3 sync "$local_path" "s3://$BUCKET/$prefix/" --storage-class "$STORAGE_CORPUS" --only-show-errors
  n_sync=$((n_sync+1))
}

# copy one meta file → run-stamped key. sqlite gets a consistent snapshot first.
do_copy() {
  local local_path="$1" tool="$2" file="$3" local_rel="$4"
  local artifact="$tool/meta/$file" key="$tool/meta/$RUN_ID/$file"
  local sig; sig="$("$PY" "$SCRIPT_DIR/_catalog.py" content-sig "$local_path")"
  if [[ $DRY_RUN -eq 1 ]]; then
    printf '  copy  %-11s %8s  → s3://%s/%s\n' "$STORAGE_META" "$(human_size "$local_path")" "$BUCKET" "$key" >&2
    n_copy=$((n_copy+1)); return
  fi
  local latest prev_sig prev_key
  latest="$(catalog_latest "$artifact")"; prev_sig="${latest%%$'\t'*}"; prev_key="${latest#*$'\t'}"
  if [[ -n "$prev_sig" && "$prev_sig" == "$sig" && -n "$prev_key" ]]; then
    log "dedup $local_rel  (unchanged → $prev_key)"
    emit_record "$artifact" "$tool" "meta" "$file" "$prev_key" "" "0" "$sig" "$STORAGE_META" "true" "" ""
    n_dedup=$((n_dedup+1)); return
  fi
  local src="$local_path" tmp=""
  if [[ "$local_path" == *.sqlite || "$local_path" == *.db ]] && command -v sqlite3 >/dev/null; then
    tmp="$(mktemp -t ragbak.XXXXXX)"
    sqlite3 "$local_path" ".backup '$tmp'" && src="$tmp"
  fi
  log "copy $(human_size "$local_path")  $local_rel  → $RUN_ID"
  aws s3 cp "$src" "s3://$BUCKET/$key" --storage-class "$STORAGE_META" \
    --metadata "localrel=$local_rel,kind=file,run_id=$RUN_ID" --only-show-errors
  local sha objsize version
  sha="$(sha256sum "$src" | cut -d' ' -f1)"
  objsize="$(aws s3api head-object --bucket "$BUCKET" --key "$key" --query 'ContentLength' --output text 2>/dev/null || echo 0)"
  version="$(aws s3api head-object --bucket "$BUCKET" --key "$key" --query 'VersionId' --output text 2>/dev/null || echo '')"
  [[ "$version" == "None" ]] && version=""
  [[ -n "$tmp" ]] && rm -f "$tmp" || true
  tag_object "$key" "$tool" "meta" "$file" "$sha"
  emit_record "$artifact" "$tool" "meta" "$file" "$key" "$sha" "$objsize" "$sig" "$STORAGE_META" "false" "" "$version"
  n_copy=$((n_copy+1))
}

while IFS=$'\t' read -r tool category method s3_subpath local_path local_rel; do
  [[ -n "$tool" ]] || continue
  artifact="$tool/$s3_subpath"
  match_only "$artifact" || continue
  name="$(basename "$s3_subpath")"
  case "$method" in
    tar)  do_tar  "$local_path" "$tool" "$category" "$name" "$s3_subpath" "$local_rel" ;;
    sync) do_sync "$local_path" "$tool" "$s3_subpath" ;;
    copy) do_copy "$local_path" "$tool" "$name" "$local_rel" ;;
    *) die "unknown method: $method" ;;
  esac
done < <(manifest_jobs "${TOOL_ARGS[@]}")

# --- finalize the run manifest (the self-contained source of truth) --------
if [[ $DRY_RUN -eq 0 && -s "$RUN_NDJSON" ]]; then
  RUN_JSON="$(mktemp -t ragbak-runjson.XXXXXX)"
  if "$PY" "$SCRIPT_DIR/_catalog.py" finalize < "$RUN_NDJSON" > "$RUN_JSON"; then
    aws s3 cp "$RUN_JSON" "s3://$BUCKET/_catalog/runs/$RUN_ID.json" \
      --storage-class STANDARD --content-type application/json --only-show-errors \
      && log "catalog → s3://$BUCKET/_catalog/runs/$RUN_ID.json"
  fi
  rm -f "$RUN_JSON"
fi

human_bytes() { awk -v b="$1" 'BEGIN{u="B KB MB GB TB";split(u,a," ");i=1;while(b>=1024&&i<5){b/=1024;i++}printf "%.1f %s",b,a[i]}'; }
SUFFIX=""; [[ $DRY_RUN -eq 1 ]] && SUFFIX=" (dry-run)"
log "done — ${n_tar} tarballs, ${n_dedup} deduped, ${n_sync} synced trees, ${n_copy} meta files; ~$(human_bytes "$bytes") uncompressed${SUFFIX}"
