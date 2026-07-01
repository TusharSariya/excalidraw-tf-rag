#!/usr/bin/env python3
"""Catalog JSON helpers for the backup harness.

The bash harness owns every `aws` call (auth, retries, env). This helper owns
only the JSON shape-shifting that is painful in bash:

    make-record    CATREC_<FIELD> env vars → flat record JSON (stdout)
    to-item        plain record JSON (stdin) → DynamoDB typed item JSON (stdout)
    extract-latest `aws dynamodb query` output (stdin) → "content_sig\ts3_key" of
                   the newest row (empty line if none) — drives dedup
    content-sig    a directory or file path (argv) → a fast content signature
                   (sha256 over sorted relpath+size+mtime); portable Mac/Linux
                   (BSD find has no -printf), metadata-only so it is cheap
    finalize       NDJSON of records (stdin) → pretty run-manifest JSON (stdout)

A "record" is the flat dict the harness builds per artifact:

    {
      "artifact": "<tool>/<category>/<name>",   # DynamoDB PK (run-id-free)
      "run_id":   "<UTC>-<host>",               # DynamoDB SK
      "tool", "category", "name",
      "s3_bucket", "s3_key", "s3_version_id",
      "sha256", "size_bytes" (int), "content_sig",
      "storage_class", "deduped" (bool),
      "host", "tool_git_sha", "created_at",
      "provenance"  # JSON string blob (ingest_status etc.), may be ""
    }

Restore never needs DynamoDB — the run-manifest JSON written to
`_catalog/runs/<run-id>.json` is the self-contained source of truth. This table
is a convenience index. Requires Python >= 3.11.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

# Keys stored as DynamoDB Number (N) rather than String (S).
_NUMERIC = {"size_bytes"}
# Keys stored as DynamoDB Bool.
_BOOL = {"deduped"}
# The flat record shape, in a stable order. Built from CATREC_<UPPER> env vars by
# `make-record` so the harness never hand-assembles JSON in bash.
_RECORD_FIELDS = (
    "artifact",
    "run_id",
    "tool",
    "category",
    "name",
    "s3_bucket",
    "s3_key",
    "s3_version_id",
    "sha256",
    "size_bytes",
    "content_sig",
    "storage_class",
    "deduped",
    "host",
    "tool_git_sha",
    "created_at",
    "provenance",
)


def cmd_make_record() -> int:
    """Build a flat record JSON (stdout) from CATREC_<FIELD> environment vars.
    Empty/unset vars are omitted; size_bytes→int, deduped→bool."""
    rec: dict = {}
    for field in _RECORD_FIELDS:
        raw = os.environ.get("CATREC_" + field.upper())
        if raw is None or raw == "":
            continue
        if field in _NUMERIC:
            rec[field] = int(raw)
        elif field in _BOOL:
            rec[field] = raw.strip().lower() in ("1", "true", "yes")
        else:
            rec[field] = raw
    json.dump(rec, sys.stdout, separators=(",", ":"))
    return 0


def _to_attr(key: str, value) -> dict:
    if key in _BOOL or isinstance(value, bool):
        return {"BOOL": bool(value)}
    if key in _NUMERIC or isinstance(value, int):
        return {"N": str(value)}
    # Everything else is a string. DynamoDB rejects empty strings is FALSE since
    # 2020 — empty strings are allowed for non-key attributes — but key
    # attributes must be non-empty; the harness guarantees artifact/run_id.
    return {"S": "" if value is None else str(value)}


def cmd_to_item() -> int:
    record = json.load(sys.stdin)
    item = {k: _to_attr(k, v) for k, v in record.items() if v is not None}
    json.dump(item, sys.stdout, separators=(",", ":"))
    return 0


def cmd_extract_latest() -> int:
    """Read `aws dynamodb query ... --scan-index-forward false --limit 1` output
    and print "content_sig\\ts3_key" of the newest row (empty line if no rows).
    The harness compares content_sig to decide dedup; on a hit it reuses s3_key."""
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.stdout.write("\t")  # no/garbled output → no prior row, no dedup
        return 0
    items = data.get("Items") or []
    if not items:
        sys.stdout.write("\t")
        return 0
    row = items[0]
    sig = row.get("content_sig", {}).get("S", "")
    key = row.get("s3_key", {}).get("S", "")
    sys.stdout.write(f"{sig}\t{key}")
    return 0


def cmd_content_sig() -> int:
    """Fast metadata-only signature of a path: sha256 over sorted
    "relpath\\0size\\0mtime_ns" for every regular file. Changes when any file is
    added/removed/resized/retouched, so it is a reliable "did this artifact
    change since the last backup?" check without reading file contents."""
    if len(sys.argv) < 3:
        sys.exit("usage: _catalog.py content-sig <path>")
    root = sys.argv[2]
    h = hashlib.sha256()
    entries: list[tuple[str, int, int]] = []
    if os.path.isfile(root):
        st = os.stat(root)
        entries.append((os.path.basename(root), st.st_size, st.st_mtime_ns))
    else:
        for dirpath, _dirs, files in os.walk(root):
            for fn in files:
                fp = os.path.join(dirpath, fn)
                try:
                    st = os.stat(fp)
                except OSError:
                    continue
                rel = os.path.relpath(fp, root)
                entries.append((rel, st.st_size, st.st_mtime_ns))
    for rel, size, mtime in sorted(entries):
        h.update(rel.encode("utf-8", "surrogatepass"))
        h.update(b"\0")
        h.update(str(size).encode())
        h.update(b"\0")
        h.update(str(mtime).encode())
        h.update(b"\n")
    sys.stdout.write(h.hexdigest())
    return 0


def cmd_sqlite_sig() -> int:
    """Logical signature of a sqlite db: sha256 over its sorted `.dump` SQL.

    A consistent backup (`sqlite3 .backup` / `conn.backup()`) is byte-different
    from the source by design — page layout, freelist and rowid gaps differ even
    though every row is identical. So a raw byte/content-sig comparison gives a
    false mismatch when drilling a backed-up db. Comparing the logical dump
    instead proves the two databases hold the same data, which is what "is this
    backup good?" actually means. Uses only the stdlib sqlite3 module so it works
    on machines without the sqlite3 CLI (the desktop has none)."""
    import sqlite3

    if len(sys.argv) < 3:
        sys.exit("usage: _catalog.py sqlite-sig <path>")
    path = sys.argv[2]
    # Open read-only via URI so we never create/modify the file being checked.
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        ok = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if ok != "ok":
            sys.exit(f"sqlite-sig: integrity_check failed for {path}: {ok}")
        h = hashlib.sha256()
        for line in conn.iterdump():
            h.update(line.encode("utf-8", "surrogatepass"))
            h.update(b"\n")
    finally:
        conn.close()
    sys.stdout.write(h.hexdigest())
    return 0


def cmd_finalize() -> int:
    """NDJSON records (stdin) → one pretty run-manifest object (stdout)."""
    records = [json.loads(line) for line in sys.stdin if line.strip()]
    run_id = records[0]["run_id"] if records else None
    manifest = {
        "schema": 1,
        "run_id": run_id,
        "host": records[0].get("host") if records else None,
        "created_at": records[0].get("created_at") if records else None,
        "artifact_count": len(records),
        "artifacts": records,
    }
    json.dump(manifest, sys.stdout, indent=2, sort_keys=True)
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        sys.exit("usage: _catalog.py {to-item|extract-sig|finalize}")
    cmd = sys.argv[1]
    if cmd == "make-record":
        return cmd_make_record()
    if cmd == "to-item":
        return cmd_to_item()
    if cmd == "extract-latest":
        return cmd_extract_latest()
    if cmd == "content-sig":
        return cmd_content_sig()
    if cmd == "sqlite-sig":
        return cmd_sqlite_sig()
    if cmd == "finalize":
        return cmd_finalize()
    sys.exit(f"error: unknown subcommand {cmd!r}")


if __name__ == "__main__":
    raise SystemExit(main())
