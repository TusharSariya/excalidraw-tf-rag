from __future__ import annotations

import difflib
import json
import logging
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from graph_layout_rag.paths import MANIFEST_PATH

log = logging.getLogger(__name__)

ManifestStatus = Literal["ok", "metadata_only", "failed"]
DocumentKind = Literal["paper", "implementation"]

# Below this token/char similarity ratio, two non-empty titles are treated as
# referring to different works (not case/punctuation/truncation variants of
# the same one). Chosen empirically: truncated-title pairs and case/punct
# variants score well above this, unrelated-paper pairs well below it.
TITLE_SIMILARITY_THRESHOLD = 0.5


class ManifestItem(BaseModel):
    id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    source: str
    url: str
    localPath: str | None = None
    contentType: str | None = None
    status: ManifestStatus
    tags: list[str] = Field(default_factory=list)
    sha256: str | None = None
    doi: str | None = None
    abstract: str | None = None
    externalIds: dict[str, str] = Field(default_factory=dict)
    discoverySources: list[str] = Field(default_factory=list)
    sourceUrls: list[str] = Field(default_factory=list)
    abstractSource: str | None = None
    documentKind: DocumentKind = "paper"
    venue: str | None = None
    arxiv_category: str | None = None
    genre: str | None = None  # e.g. OpenAlex `type` or S2 publicationType
    venue_type: str | None = (
        None  # OpenAlex primary_location.source.type (journal/conference/repository)
    )
    oa_version: str | None = (
        None  # OpenAlex primary_location.version (publishedVersion/acceptedVersion/submittedVersion)
    )
    is_retracted: bool = False
    # Former manifest ids this record has absorbed (via DOI/sha256 merges or
    # duplicate-collapse repairs). Read by query/identity.py's canonical
    # identity map and projected into the index's alias_doc_ids column, so
    # qrels/citation references to a swallowed id still resolve here. Without
    # this, a merge that re-keys a record silently orphans every qrels
    # judgment for it.
    aliasDocIds: list[str] = Field(default_factory=list)


class Manifest(BaseModel):
    version: int = 1
    updatedAt: str = ""
    items: list[ManifestItem] = Field(default_factory=list)


def slug_id(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:80]


def _normalize_title(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def titles_similar(
    a: str | None, b: str | None, threshold: float = TITLE_SIMILARITY_THRESHOLD
) -> bool:
    """True if two titles look like the same work (or one can't be checked).

    Empty/missing titles are treated as "can't compare" -> similar, so callers
    only get a conflict signal when both sides actually have a title to
    compare. Comparison is on lowercased, punctuation-stripped text via
    difflib's ratio, which tolerates case/punctuation differences and simple
    truncation (a common-prefix substring still scores highly).
    """
    if not a or not b:
        return True
    na, nb = _normalize_title(a), _normalize_title(b)
    if not na or not nb:
        return True
    return difflib.SequenceMatcher(None, na, nb).ratio() >= threshold


def load_manifest() -> Manifest:
    if not MANIFEST_PATH.exists():
        return Manifest(updatedAt=_now_iso())
    raw = MANIFEST_PATH.read_text(encoding="utf-8")
    return Manifest.model_validate(json.loads(raw))


def save_manifest(manifest: Manifest) -> None:
    manifest.updatedAt = _now_iso()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST_PATH.with_suffix(".json.tmp")
    tmp.write_text(
        f"{manifest.model_dump_json(indent=2)}\n",
        encoding="utf-8",
    )
    os.replace(tmp, MANIFEST_PATH)


def merge_items(
    existing: ManifestItem, incoming: ManifestItem, *, check_title_conflict: bool = True
) -> ManifestItem:
    """Merge provider records without losing a stronger previously discovered field.

    ``check_title_conflict`` guards against merging two different papers whose
    records were matched by an inferred key (DOI, sha256) that turned out to
    be a collision (e.g. a synthesized arXiv DOI shared by unrelated works).
    When both sides have a non-empty title and the titles are dissimilar, the
    identity-bearing fields (title/abstract/authors) are kept from ``existing``
    regardless of which record would otherwise be preferred by status, and the
    conflict is logged. Callers that matched on an authoritative key (exact
    manifest id) should pass ``check_title_conflict=False`` since an id match
    is not an inferred/ambiguous signal.
    """
    status_rank = {"failed": 0, "metadata_only": 1, "ok": 2}
    preferred = (
        incoming if status_rank[incoming.status] >= status_rank[existing.status] else existing
    )
    fallback = existing if preferred is incoming else incoming
    merged = preferred.model_copy(deep=True)

    title_conflict = (
        check_title_conflict
        and bool(existing.title)
        and bool(incoming.title)
        and not titles_similar(existing.title, incoming.title)
    )
    if title_conflict:
        log.warning(
            "manifest merge conflict: dissimilar titles for shared key "
            "(doi=%r sha256=%r) — existing=%r incoming=%r; keeping existing "
            "title/abstract/authors",
            existing.doi or incoming.doi,
            existing.sha256 or incoming.sha256,
            existing.title,
            incoming.title,
        )
        merged.title = existing.title
        merged.abstract = existing.abstract
        merged.authors = list(existing.authors)

    backfill_fields = [
        "title",
        "authors",
        "year",
        "url",
        "localPath",
        "contentType",
        "sha256",
        "doi",
        "abstract",
        "abstractSource",
        "venue",
        "arxiv_category",
        "genre",
        "venue_type",
        "oa_version",
    ]
    if title_conflict:
        backfill_fields = [f for f in backfill_fields if f not in ("title", "abstract", "authors")]

    for field in backfill_fields:
        if not getattr(merged, field):
            setattr(merged, field, getattr(fallback, field))

    merged.is_retracted = existing.is_retracted or incoming.is_retracted
    # Union alias ids and both record ids: whichever id does not survive the
    # merge (the caller stamps merged.id afterwards) must remain reachable as
    # an alias. upsert_item removes the surviving id from the list.
    merged.aliasDocIds = sorted(
        (set(existing.aliasDocIds) | set(incoming.aliasDocIds) | {existing.id, incoming.id}) - {""}
    )
    merged.tags = sorted(set(existing.tags) | set(incoming.tags))
    merged.discoverySources = sorted(
        set(existing.discoverySources)
        | set(incoming.discoverySources)
        | {existing.source, incoming.source}
    )
    merged.sourceUrls = sorted(
        set(existing.sourceUrls)
        | set(incoming.sourceUrls)
        | {u for u in (existing.url, incoming.url) if u}
    )
    merged.externalIds = {**existing.externalIds, **incoming.externalIds}
    if merged.doi:
        merged.externalIds.setdefault("DOI", merged.doi)
    return merged


def _is_title_collision(existing: ManifestItem, item: ManifestItem) -> bool:
    """True when both records have a title and the titles are dissimilar.

    Used to distinguish a genuine match (same paper, different provider) from
    a false match on an inferred key (DOI, sha256) that happens to collide
    across two unrelated papers.
    """
    return (
        bool(existing.title) and bool(item.title) and not titles_similar(existing.title, item.title)
    )


def upsert_item(manifest: Manifest, item: ManifestItem) -> None:
    for idx, existing in enumerate(manifest.items):
        id_match = existing.id == item.id
        same_doi = bool(
            existing.doi
            and item.doi
            and existing.doi.lower().removeprefix("https://doi.org/")
            == item.doi.lower().removeprefix("https://doi.org/")
        )
        same_sha256 = bool(existing.sha256 and item.sha256 and existing.sha256 == item.sha256)

        if id_match:
            merged = merge_items(existing, item, check_title_conflict=False)
            merged.id = existing.id
            merged.aliasDocIds = [a for a in merged.aliasDocIds if a != merged.id]
            manifest.items[idx] = merged
            return

        if same_doi:
            if _is_title_collision(existing, item):
                log.warning(
                    "manifest DOI collision: doi=%r matches existing id=%r "
                    "(title=%r) but incoming id=%r has a dissimilar title "
                    "(%r) — treating as distinct items, not merging",
                    existing.doi or item.doi,
                    existing.id,
                    existing.title,
                    item.id,
                    item.title,
                )
                continue
            merged = merge_items(existing, item)
            merged.id = existing.id
            merged.aliasDocIds = [a for a in merged.aliasDocIds if a != merged.id]
            manifest.items[idx] = merged
            return

        if same_sha256:
            if _is_title_collision(existing, item):
                log.warning(
                    "manifest sha256 collision: identical file hash %r shared "
                    "by existing id=%r (title=%r) and incoming id=%r "
                    "(title=%r) with dissimilar titles — not auto-merging, "
                    "leaving as distinct items for manual review",
                    existing.sha256 or item.sha256,
                    existing.id,
                    existing.title,
                    item.id,
                    item.title,
                )
                continue
            merged = merge_items(existing, item)
            merged.id = existing.id
            merged.aliasDocIds = [a for a in merged.aliasDocIds if a != merged.id]
            manifest.items[idx] = merged
            return

    manifest.items.append(item)


def manifest_by_id(manifest: Manifest) -> dict[str, ManifestItem]:
    return {item.id: item for item in manifest.items}


def relative_local_path(abs_path: Path) -> str:
    from graph_layout_rag.paths import PKG_ROOT

    return str(abs_path.relative_to(PKG_ROOT))


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
