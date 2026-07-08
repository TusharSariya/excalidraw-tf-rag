#!/usr/bin/env bash
# Fast, local pre-push guard for v3/supabase/migrations/ — no Docker, no network.
# Catches the two classes of bug that caused a real mess this session: a
# malformed/duplicate migration filename, and a table created without RLS
# (Supabase auto-exposes every `public` schema table via its Data API).
set -euo pipefail

MIGRATIONS_DIR="v3/supabase/migrations"
[ -d "$MIGRATIONS_DIR" ] || exit 0

fail=0
seen_timestamps=""

for f in "$MIGRATIONS_DIR"/*.sql; do
    [ -e "$f" ] || continue
    base=$(basename "$f")

    if ! [[ "$base" =~ ^[0-9]{14}_[a-z0-9_]+\.sql$ ]]; then
        echo "ERROR: $f does not match required migration filename pattern <14-digit-timestamp>_<name>.sql"
        fail=1
        continue
    fi

    ts="${base%%_*}"
    if [[ " $seen_timestamps " == *" $ts "* ]]; then
        echo "ERROR: duplicate migration timestamp $ts (file: $f)"
        fail=1
    fi
    seen_timestamps="$seen_timestamps $ts"

    while IFS= read -r table; do
        [ -n "$table" ] || continue
        if ! grep -qiE "alter[[:space:]]+table[[:space:]]+(public\.)?${table}[[:space:]]+enable[[:space:]]+row[[:space:]]+level[[:space:]]+security" "$f"; then
            echo "ERROR: $f creates table '$table' without enabling row level security in the same file"
            fail=1
        fi
    done < <(grep -oiE "create[[:space:]]+table[[:space:]]+public\.[a-z0-9_]+" "$f" | awk '{print $NF}' | sed 's/^public\.//')
done

exit $fail
