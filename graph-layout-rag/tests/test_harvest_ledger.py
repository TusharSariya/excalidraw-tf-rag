from graph_layout_rag.harvest.ledger import (
    classify_outcome,
    init_db,
    log_attempt,
    query_attempts,
    summary,
)


def test_classify_outcome():
    assert classify_outcome(status=429) == ("rate_limited", True)
    assert classify_outcome(status=403) == ("forbidden", False)
    assert classify_outcome(status=200, ok=True) == ("ok", False)
    assert classify_outcome(status=200, reason="non-PDF") == ("not_pdf", False)


def test_ledger_log_and_query(tmp_path, monkeypatch):
    db_path = tmp_path / "harvest.db"
    monkeypatch.setattr("graph_layout_rag.harvest.ledger.HARVEST_DB_PATH", db_path)
    init_db()
    log_attempt(
        url="https://example.com/paper.pdf",
        outcome="rate_limited",
        transient=True,
        doc_id="test-doc",
        doi="10.1234/test",
        http_status=429,
    )
    rows = query_attempts(transient=True, limit=10)
    assert len(rows) == 1
    assert rows[0]["outcome"] == "rate_limited"
    rep = summary()
    assert rep["total_attempts"] == 1
    assert rep["by_outcome"]["rate_limited"] == 1


def test_log_attempt_without_prior_init_db(tmp_path, monkeypatch):
    """log_attempt must self-init the ledger schema.

    Regression for a bug where harvest paths that called log_attempt()
    before any init_db() left the `download_attempts` table missing,
    raising sqlite3.OperationalError. A broad `except Exception` around
    the download call in topic_seeds._download_pdf_seed then caught that
    error and deleted the PDF it had just successfully downloaded.
    """
    db_path = tmp_path / "harvest.db"
    monkeypatch.setattr("graph_layout_rag.harvest.ledger.HARVEST_DB_PATH", db_path)
    assert not db_path.exists()
    log_attempt(
        url="https://example.com/paper.pdf",
        outcome="ok",
        transient=False,
        doc_id="test-doc",
        http_status=200,
    )
    rows = query_attempts(limit=10)
    assert len(rows) == 1
    assert rows[0]["outcome"] == "ok"
