import hashlib
import sqlite3

from rag_literature_rag.catalog.taxonomy import RAG_CATEGORIES
from rag_literature_rag.harvest.topic_seeds import (
    TOPIC_DOI_SEEDS,
    TOPIC_METADATA_SEEDS,
    TOPIC_PDF_SEEDS,
    _download_pdf_seed,
    harvest_topic_seeds,
)


def test_topic_pdf_seed_ids():
    ids = {s["id"] for s in TOPIC_PDF_SEEDS}
    assert "arxiv-2005-11401" in ids
    assert "arxiv-2004-04906" in ids
    assert "arxiv-2004-12832" in ids
    assert "arxiv-2312-10997" in ids


def test_topic_doi_seed_count():
    assert len(TOPIC_DOI_SEEDS) >= 40
    dois = {s["doi"] for s in TOPIC_DOI_SEEDS}
    assert "10.48550/arXiv.2310.11511" in dois  # Self-RAG
    assert "10.48550/arXiv.2404.16130" in dois  # GraphRAG
    assert "10.48550/arXiv.2603.07379" in dois  # SoK Agentic RAG


def test_topic_seed_tags_use_rag_taxonomy():

    allowed = set(RAG_CATEGORIES) | {
        "topic-seed",
        "late-interaction",
        "parametric-memory",
        "pretraining",
        "reader",
        "generation",
        "taxonomy",
        "reflection",
        "evaluator",
        "hierarchical-index",
        "iterative-retrieval",
        "query-rewriting",
        "dual-tuning",
        "compression",
        "noise",
        "filtering",
        "adaptive",
        "uncertainty",
        "unified",
        "draft-retrieval",
        "ablation",
        "community",
        "ppr",
        "dual-level",
        "heterogeneous",
        "query-aware",
        "reasoning",
        "sok",
        "trust",
        "interfaces",
        "repair",
        "planning",
        "framework",
        "optimization",
        "robustness",
        "dynamic",
        "multi-hop",
        "bm25",
        "contextual",
        "citation",
        "comparison",
        "contextual-retrieval",
        "implementer-guide",
        "embeddings",
        "context",
        "metrics",
    }
    for spec in [*TOPIC_PDF_SEEDS, *TOPIC_DOI_SEEDS, *TOPIC_METADATA_SEEDS]:
        for tag in spec.get("tags", []):
            assert tag in allowed, f"unknown tag {tag!r} on {spec.get('id') or spec.get('doi')}"


def test_metadata_seeds_present():
    ids = {s["id"] for s in TOPIC_METADATA_SEEDS}
    assert "anthropic-contextual-retrieval-cookbook" in ids
    assert "llamaindex-advanced-rag" in ids


def test_harvest_topic_seeds_dry_run():
    items = harvest_topic_seeds(dry_run=True)
    ids = {i.id for i in items}
    assert "arxiv-2005-11401" in ids
    assert "anthropic-contextual-retrieval-cookbook" in ids
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
    import rag_literature_rag.harvest.topic_seeds as topic_seeds

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
