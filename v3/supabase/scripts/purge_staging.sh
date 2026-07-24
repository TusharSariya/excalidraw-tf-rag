#!/usr/bin/env bash
#
# Purge harvest state (seeds + everything downstream + the harvest queue messages)
# from a Supabase Postgres database. See purge_staging.sql for the exact inventory.
# Only the queue *mailboxes* are emptied — the queues themselves, queue_subscriptions,
# and harvest_generation (the fan-out control-plane) are left intact.
#
# Connection is resolved in this order:
#   1. $DATABASE_URL                     — a full libpq URL, used verbatim
#   2. SUPABASE_POOLER_URL + SUPABASE_STAGING_PW from v3/harvest/.env
#      The URL's [YOUR-PASSWORD] placeholder is ignored; the password is passed
#      to psql via PGPASSWORD and is never interpolated into any printed string.
#
# Usage:
#   ./purge_staging.sh              # print target + row counts, then confirm
#   ./purge_staging.sh --yes        # skip the confirmation prompt (automation)
#   ./purge_staging.sh --dry-run    # print target + row counts only; no changes
#
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sql_file="$here/purge_staging.sql"
env_file="$here/../../harvest/.env"

# Keep in sync with the TRUNCATE list in purge_staging.sql.
tables=(
  seeds
  source_arxiv source_openalex source_semantic_scholar
  harvest_dispatch
  s2_paper s2_author s2_paper_author s2_reference s2_citation s2_embedding
)

mode="prompt"
case "${1:-}" in
  --yes|-y)      mode="yes" ;;
  --dry-run|-n)  mode="dry" ;;
  "")            ;;
  *) echo "unknown argument: $1 (use --yes or --dry-run)" >&2; exit 2 ;;
esac

# --- resolve connection ------------------------------------------------------
read_env() {  # key file -> value with surrounding quotes stripped
  local k="$1" f="$2"
  [ -f "$f" ] || return 0
  sed -n "s/^${k}=//p" "$f" | tail -1 | sed -e 's/^["'\'']//' -e 's/["'\'']$//'
}

pgpassword=""
if [ -n "${DATABASE_URL:-}" ]; then
  conn_url="$DATABASE_URL"
else
  conn_url="$(read_env SUPABASE_POOLER_URL "$env_file")"
  pgpassword="$(read_env SUPABASE_STAGING_PW "$env_file")"
  if [ -z "$conn_url" ]; then
    echo "No \$DATABASE_URL set, and SUPABASE_POOLER_URL not found in $env_file" >&2
    exit 1
  fi
fi

# Parse user/host/port/db from the URL, discarding any password or placeholder.
noproto="${conn_url#*://}"
userinfo="${noproto%%@*}"
hostpart="${noproto#*@}"
user="${userinfo%%:*}"
db="${hostpart##*/}"; db="${db%%\?*}"
hostport="${hostpart%%/*}"
host="${hostport%%:*}"
port="${hostport##*:}"; [ "$port" = "$host" ] && port=5432
ref="${user#postgres.}"          # Supabase pooler user is postgres.<project-ref>

# If a real (non-placeholder) password rode inline in the URL, honour it.
if [ -z "$pgpassword" ]; then
  cand="${userinfo#*:}"
  case "$cand" in
    ""|"$userinfo"|*"[YOUR-PASSWORD]"*) ;;
    *) pgpassword="$cand" ;;
  esac
fi

export PGPASSWORD="$pgpassword"
psql_base=(-h "$host" -p "$port" -U "$user" -d "$db" -v ON_ERROR_STOP=1 --no-psqlrc)

echo "Target Supabase database:"
echo "  project ref : ${ref}"
echo "  host        : ${host}:${port}/${db}"
echo

# --- exact row counts (read-only) -------------------------------------------
counts_sql=""
for t in "${tables[@]}"; do
  [ -n "$counts_sql" ] && counts_sql+=" union all "
  counts_sql+="select '${t}' as table_name, count(*)::bigint as rows from public.${t}"
done
counts_sql+=" order by table_name;"

# Depth of each harvest_* pgmq queue (messages that purge_queue will clear).
queues_sql="select queue_name, (pgmq.metrics(queue_name)).queue_length as messages
            from pgmq.list_queues() where queue_name like 'harvest\_%' order by queue_name;"

show_state() {  # $1 = label
  echo "Row counts ($1):"
  psql "${psql_base[@]}" -c "$counts_sql"
  echo "Queue depths ($1):"
  psql "${psql_base[@]}" -c "$queues_sql"
  echo
}

show_state "pre-purge"

if [ "$mode" = "dry" ]; then
  echo "--dry-run: connection OK, no changes made."
  exit 0
fi

# --- confirm -----------------------------------------------------------------
if [ "$mode" = "prompt" ]; then
  echo "This will PERMANENTLY TRUNCATE every table above AND purge every harvest_*"
  echo "queue message in project '${ref}'. The queues, queue_subscriptions, and"
  echo "harvest_generation are kept (only the messages are cleared)."
  printf 'Type the project ref (%s) to proceed, anything else to abort: ' "$ref"
  read -r reply
  if [ "$reply" != "$ref" ]; then
    echo "Confirmation did not match; aborting. Nothing was changed."
    exit 1
  fi
fi

# --- purge -------------------------------------------------------------------
echo "Purging..."
psql "${psql_base[@]}" -f "$sql_file"
echo
show_state "post-purge"
echo "Done. Left intact: the pgmq queues themselves, queue_subscriptions, harvest_generation."
