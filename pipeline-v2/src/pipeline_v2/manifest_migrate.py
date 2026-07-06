"""v1 manifest.json -> pipeline-v2 ``manifest_item`` table, and back.

Deliberately dict-level: this module does NOT import v1's
``graph_layout_rag.manifest`` (or the rag-literature-rag equivalent) so
pipeline-v2 stays decoupled from v1 package internals per the coexistence
directive (v1 packages are read-only reference for this phase). The v1
``ManifestItem``/``Manifest`` pydantic field shapes were read (not imported)
from ``graph-layout-rag/src/graph_layout_rag/manifest.py`` to derive the
extracted key columns and the export shape below.

v1 manifest shape (for reference, not enforced by a schema here — payload is
stored as opaque jsonb, so an unknown/extra field round-trips untouched):

    {
      "version": 1,
      "updatedAt": "<iso8601>",
      "items": [
        {"id": ..., "title": ..., "authors": [...], "year": ...,
         "source": ..., "url": ..., "localPath": ..., "contentType": ...,
         "status": "ok"|"metadata_only"|"failed", "tags": [...],
         "sha256": ..., "doi": ..., "abstract": ..., "externalIds": {...},
         "discoverySources": [...], "sourceUrls": [...],
         "abstractSource": ..., "documentKind": "paper"|"implementation",
         "venue": ..., "arxiv_category": ..., "genre": ...,
         "venue_type": ..., "oa_version": ..., "is_retracted": ...},
        ...
      ]
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline_v2 import db


@dataclass(frozen=True)
class DriftReport:
    """Result of comparing a re-exported manifest against its own canonical input."""

    drift: bool
    input_item_count: int
    exported_item_count: int
    missing_ids: list[str]  # in input, not in export
    extra_ids: list[str]  # in export, not in input
    changed_ids: list[str]  # present in both, payload differs
    canonical_input: str
    canonical_export: str

    def summary(self) -> str:
        if not self.drift:
            return f"no drift: {self.input_item_count} items round-tripped byte-identical"
        parts = [f"DRIFT DETECTED ({self.input_item_count} in, {self.exported_item_count} out)"]
        if self.missing_ids:
            parts.append(f"missing from export: {self.missing_ids}")
        if self.extra_ids:
            parts.append(f"extra in export: {self.extra_ids}")
        if self.changed_ids:
            parts.append(f"changed payload: {self.changed_ids}")
        return "; ".join(parts)


def canonical_json(obj: Any) -> str:
    """Deterministic, sorted-keys serialization used for byte comparison."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def load_v1_manifest_json(path: Path) -> dict:
    """Parse a v1 manifest.json file into a plain dict (no v1 model import)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _extract_key_columns(item: dict) -> dict:
    return {
        "id": item["id"],
        "doi": item.get("doi"),
        "status": item.get("status"),
        "sha256": item.get("sha256"),
    }


def migrate_to_db(manifest: dict, *, corpus: str, dsn: str | None = None) -> int:
    """Upsert every item in a parsed v1 manifest dict into manifest_item.

    Returns the number of items written.
    """
    items = manifest.get("items", [])
    with db.connect(dsn) as conn:
        with conn.cursor() as cur:
            for item in items:
                key = _extract_key_columns(item)
                cur.execute(
                    """
                    INSERT INTO manifest_item (id, corpus, doi, status, sha256, payload, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (id, corpus) DO UPDATE SET
                        doi = EXCLUDED.doi,
                        status = EXCLUDED.status,
                        sha256 = EXCLUDED.sha256,
                        payload = EXCLUDED.payload,
                        updated_at = now()
                    """,
                    (
                        key["id"],
                        corpus,
                        key["doi"],
                        key["status"],
                        key["sha256"],
                        json.dumps(item),
                    ),
                )
        conn.commit()
    return len(items)


def export_from_db(*, corpus: str, dsn: str | None = None) -> dict:
    """Reconstruct a v1-shaped manifest dict from stored payloads for one corpus.

    Items are ordered by id for determinism (v1 manifest.json has no
    guaranteed item order, so sorting is required for a stable byte compare).
    """
    from psycopg.rows import dict_row

    with db.connect(dsn, autocommit=True) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT payload FROM manifest_item WHERE corpus = %s ORDER BY id",
                (corpus,),
            )
            items = [row["payload"] for row in cur.fetchall()]

    updated_at = ""
    return {"version": 1, "updatedAt": updated_at, "items": items}


def diff_manifests(input_manifest: dict, exported_manifest: dict) -> DriftReport:
    """Compare an exported manifest against its own input, key-normalized.

    ``updatedAt`` is excluded from the comparison (export does not attempt to
    reproduce it — see :func:`export_from_db`); everything else, including
    unknown/extra fields on each item, must match byte-for-byte once both
    sides are canonicalized (sorted keys, no incidental whitespace).
    """
    input_items = {item["id"]: item for item in input_manifest.get("items", [])}
    exported_items = {item["id"]: item for item in exported_manifest.get("items", [])}

    missing_ids = sorted(set(input_items) - set(exported_items))
    extra_ids = sorted(set(exported_items) - set(input_items))
    changed_ids = sorted(
        item_id
        for item_id in set(input_items) & set(exported_items)
        if canonical_json(input_items[item_id]) != canonical_json(exported_items[item_id])
    )

    canonical_input = canonical_json({"items": [input_items[i] for i in sorted(input_items)]})
    canonical_export = canonical_json(
        {"items": [exported_items[i] for i in sorted(exported_items)]}
    )

    return DriftReport(
        drift=bool(missing_ids or extra_ids or changed_ids),
        input_item_count=len(input_items),
        exported_item_count=len(exported_items),
        missing_ids=missing_ids,
        extra_ids=extra_ids,
        changed_ids=changed_ids,
        canonical_input=canonical_input,
        canonical_export=canonical_export,
    )


def export_to_json_file(manifest: dict, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(f"{json.dumps(manifest, indent=2, sort_keys=True)}\n", encoding="utf-8")
