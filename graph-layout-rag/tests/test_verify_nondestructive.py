"""Regression: verify_manifest must never demote ok items over topicality.

2026-07-04 incident: the final ``verify_manifest(downgrade=True)`` step of a
harvest run flipped 1,196 ok items to failed — and nulled their
localPath/sha256 — solely because ``is_layout_relevant`` scored their
title/abstract below threshold, even though every one had a valid PDF on
disk. Verify's downgrade authority is limited to structural PDF integrity
(missing/corrupt/truncated file); topical curation belongs exclusively to
``harvest prune`` (explicit, previewed, ``--apply --yes``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from graph_layout_rag.harvest import verify as verify_mod
from graph_layout_rag.harvest.verify import verify_manifest
from graph_layout_rag.manifest import Manifest, ManifestItem

# Clearly off-topic for the layout relevance model (a medical title), from a
# non-curated source — exactly the shape of the items demoted in the incident.
OFF_TOPIC_TITLE = "Randomized clinical trial of vaccine efficacy in cancer patients"


def _item(item_id: str, local_path: str | None, status: str = "ok") -> ManifestItem:
    return ManifestItem(
        id=item_id,
        title=OFF_TOPIC_TITLE,
        source="bibliography",
        url=f"https://doi.org/10.9999/{item_id}",
        localPath=local_path,
        status=status,
        sha256="deadbeef",
    )


@pytest.fixture()
def pdf_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    pdf_dir = tmp_path / "data" / "raw" / "pdf"
    pdf_dir.mkdir(parents=True)
    monkeypatch.setattr(verify_mod, "PKG_ROOT", tmp_path)
    monkeypatch.setattr(verify_mod, "PDF_DIR", pdf_dir)
    return tmp_path


def _write_valid_pdf(root: Path, rel: str) -> None:
    path = root / rel
    path.write_bytes(b"%PDF-1.4\n" + b"x" * verify_mod.MIN_PDF_BYTES)


def test_off_topic_ok_item_with_valid_pdf_is_never_demoted(pdf_root: Path) -> None:
    rel = "data/raw/pdf/offtopic-1.pdf"
    _write_valid_pdf(pdf_root, rel)
    item = _item("offtopic-1", rel)
    manifest = Manifest(updatedAt="2026-07-04T00:00:00Z", items=[item])

    stats = verify_manifest(manifest, downgrade=True)

    assert item.status == "ok"
    assert item.localPath == rel
    assert item.sha256 == "deadbeef"
    assert stats["downgraded"] == 0
    assert stats["valid"] == 1
    # Topicality is still *reported* so prune has a signal to act on.
    assert stats["off_topic"] == 1


def test_structurally_broken_pdf_is_still_downgraded(pdf_root: Path) -> None:
    rel = "data/raw/pdf/broken-1.pdf"
    (pdf_root / rel).write_bytes(b"<html>not a pdf</html>")
    item = _item("broken-1", rel)
    manifest = Manifest(updatedAt="2026-07-04T00:00:00Z", items=[item])

    stats = verify_manifest(manifest, downgrade=True)

    assert item.status == "failed"
    assert item.localPath is None
    assert item.sha256 is None
    assert stats["downgraded"] == 1


def test_missing_pdf_is_still_downgraded(pdf_root: Path) -> None:
    item = _item("gone-1", "data/raw/pdf/gone-1.pdf")
    manifest = Manifest(updatedAt="2026-07-04T00:00:00Z", items=[item])

    stats = verify_manifest(manifest, downgrade=True)

    assert item.status == "failed"
    assert stats["downgraded"] == 1


def test_verify_never_drops_items(pdf_root: Path) -> None:
    rel = "data/raw/pdf/keep-1.pdf"
    _write_valid_pdf(pdf_root, rel)
    items = [
        _item("keep-1", rel),
        _item("meta-1", None, status="metadata_only"),
        _item("failed-1", None, status="failed"),
    ]
    manifest = Manifest(updatedAt="2026-07-04T00:00:00Z", items=list(items))

    verify_manifest(manifest, downgrade=True)

    assert [i.id for i in manifest.items] == ["keep-1", "meta-1", "failed-1"]
