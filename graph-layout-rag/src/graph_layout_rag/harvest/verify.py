"""Manifest integrity verification — ensure ok items have valid PDFs on disk."""

from __future__ import annotations

from graph_layout_rag.harvest.log import get_logger
from graph_layout_rag.harvest.relevance import CURATED_SOURCES, is_layout_relevant
from graph_layout_rag.manifest import Manifest, ManifestItem
from graph_layout_rag.paths import PDF_DIR, PKG_ROOT

MIN_PDF_BYTES = 10_000


def _pdf_valid(item: ManifestItem) -> tuple[bool, str]:
    """Structural PDF integrity only — never a topicality judgment.

    Topical curation is ``harvest prune``'s job (explicit, previewed, gated
    behind ``--apply --yes``). Verify must not demote an item whose PDF is
    present and well-formed: doing so nulls localPath/sha256 and orphans a
    perfectly good file (2026-07-04 incident: 1,196 ok items with valid
    on-disk PDFs were flipped to failed by an off-topic check that lived
    here).
    """
    if not item.localPath:
        return False, "no localPath"
    path = PKG_ROOT / item.localPath
    if not path.exists():
        return False, "file missing"
    data = path.read_bytes()
    if not data.startswith(b"%PDF"):
        return False, "not a PDF"
    if len(data) < MIN_PDF_BYTES:
        return False, f"too small ({len(data)} bytes)"
    return True, "ok"


def _looks_off_topic(item: ManifestItem) -> bool:
    return item.source not in CURATED_SOURCES and not is_layout_relevant(item.title, item.abstract)


def verify_manifest(manifest: Manifest, *, downgrade: bool = True) -> dict[str, int]:
    """Verify ok items have valid PDFs; optionally downgrade broken entries.

    Off-topic-looking items are *counted and reported only* (``off_topic``
    stat) — never downgraded here. Use ``harvest prune`` to act on them.
    """
    log = get_logger()
    stats = {"checked": 0, "valid": 0, "downgraded": 0, "off_topic": 0}

    for item in manifest.items:
        if item.status != "ok":
            continue
        stats["checked"] += 1
        valid, reason = _pdf_valid(item)
        if valid:
            stats["valid"] += 1
            if _looks_off_topic(item):
                stats["off_topic"] += 1
                log.info("verify: %s looks off-topic (kept; use prune to act)", item.id)
            continue
        log.warning("verify: %s invalid — %s", item.id, reason)
        if downgrade:
            item.status = "failed"
            item.localPath = None
            item.sha256 = None
            stats["downgraded"] += 1

    stats.update(_orphan_stats(manifest))
    return stats


def _orphan_stats(manifest: Manifest) -> dict[str, int]:
    """PDFs on disk without a matching ok manifest entry."""
    ok_ids = {item.id for item in manifest.items if item.status == "ok"}
    orphan_pdfs = 0
    if PDF_DIR.exists():
        for path in PDF_DIR.glob("*.pdf"):
            if path.stem not in ok_ids:
                orphan_pdfs += 1
    return {"orphan_pdfs": orphan_pdfs}
