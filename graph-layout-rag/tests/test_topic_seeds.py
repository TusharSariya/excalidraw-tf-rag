import hashlib
import sqlite3

from graph_layout_rag.harvest.topic_seeds import (
    PIPELINE_LAYOUT_DOI_SEEDS,
    TOPIC_DOI_SEEDS,
    TOPIC_METADATA_SEEDS,
    TOPIC_PDF_SEEDS,
    _download_pdf_seed,
    harvest_topic_seeds,
)


def test_topic_pdf_seed_ids():
    ids = {s["id"] for s in TOPIC_PDF_SEEDS}
    assert "elk-eclipse-layout-kernel-arxiv" in ids
    assert "sander-compound-directed-graphs" in ids
    assert "stratisfimal-layout" in ids
    assert "eades-1984-spring-heuristic" in ids
    assert "hu-2005-efficient-high-quality-fdp" in ids


def test_topic_doi_seed_count():
    assert len(TOPIC_DOI_SEEDS) >= 52
    assert len(PIPELINE_LAYOUT_DOI_SEEDS) >= 19


def test_thin_category_seeds_present():
    dois = {s["doi"] for s in PIPELINE_LAYOUT_DOI_SEEDS}
    # compaction
    assert "10.1007/3-540-48777-8_23" in dois  # Klau & Mutzel optimal compaction
    assert "10.1016/S0925-7721(01)00010-4" in dois  # Patrignani complexity
    # coordinate assignment
    assert "10.7155/jgaa.00126" in dois  # Eiglsperger Sugiyama JGAA
    # packing
    assert "10.1007/3-540-45848-4_30" in dois  # polyomino packing
    # overlap removal
    assert "10.1007/978-3-642-00219-9_20" in dois  # PRISM
    assert "10.1007/978-3-319-50106-2_3" in dois  # growing a tree


def test_canonical_seed_dois_present():
    dois = {s["doi"] for s in TOPIC_DOI_SEEDS}
    # tree layout, orthogonal/TSM, force-directed evaluation, surveys
    assert "10.1007/3-540-36151-0_32" in dois  # Buchheim tidy tree
    assert "10.1137/0216030" in dois  # Tamassia bend minimization
    assert "10.7155/jgaa.00154" in dois  # Hachul & Jünger experimental study
    assert "10.1016/0925-7721(94)00014-X" in dois  # Di Battista annotated bibliography
    # adaptagrams / libavoid / libcola
    assert "10.1007/978-3-642-31223-6_8" in dois  # Orthogonal Hyperedge Routing
    assert "10.1109/TVCG.2008.130" in dois  # Cooperative constraint layout


def test_thesis_metadata_seeds_present():
    ids = {s["id"] for s in TOPIC_METADATA_SEEDS}
    assert "thesis-ruegg-sugiyama-prescribed-areas" in ids
    assert "thesis-schulze-layered-port-constraints" in ids
    assert "thesis-klau-orthogonal-placement" in ids


def test_research_thread_metadata():
    ids = {s["id"] for s in TOPIC_METADATA_SEEDS}
    assert "research-thread-compaction" in ids
    assert "research-thread-packing" in ids
    assert "research-thread-overlap" in ids


def test_harvest_topic_seeds_dry_run():
    items = harvest_topic_seeds(dry_run=True)
    ids = {i.id for i in items}
    assert "elk-eclipse-layout-kernel-arxiv" in ids
    assert "mermaid-layouts-docs" in ids
    assert "research-thread-compaction" in ids
    assert any(i.tags and "topic-seed" in i.tags for i in items)


def test_download_pdf_seed_survives_ledger_write_failure(tmp_path, monkeypatch):
    """A ledger-logging failure after a successful write must not delete the PDF.

    Regression for a bug where download_to_file writes the PDF to disk and
    only then logs the attempt to the harvest sqlite ledger. If that logging
    step raises (e.g. sqlite3.OperationalError: no such table, from a ledger
    that hasn't been initialized), download_to_file propagates the exception,
    and _download_pdf_seed's broad `except Exception` used to delete the
    just-downloaded, perfectly valid PDF. It must now survive.
    """
    import graph_layout_rag.harvest.topic_seeds as topic_seeds

    monkeypatch.setattr(topic_seeds, "PDF_DIR", tmp_path)
    monkeypatch.setattr(topic_seeds, "relative_local_path", lambda path: str(path))
    pdf_bytes = b"%PDF-1.4\n" + b"0" * 20_000

    def fake_download_to_file(dest, url, **kwargs):
        # Mirrors the real download_to_file: write the file first, then blow
        # up as if the ledger's log_attempt() call failed.
        dest.write_bytes(pdf_bytes)
        raise sqlite3.OperationalError("no such table: download_attempts")

    monkeypatch.setattr(topic_seeds, "download_to_file", fake_download_to_file)

    spec = {
        "id": "regression-ledger-failure",
        "title": "Regression Ledger Failure",
        "url": "https://example.com/paper.pdf",
    }
    item = _download_pdf_seed(spec, dry_run=False)

    dest = tmp_path / "regression-ledger-failure.pdf"
    assert dest.exists(), "downloaded PDF must survive a ledger-logging failure"
    assert dest.read_bytes() == pdf_bytes
    assert item.status == "ok"
    assert item.sha256 == hashlib.sha256(pdf_bytes).hexdigest()
