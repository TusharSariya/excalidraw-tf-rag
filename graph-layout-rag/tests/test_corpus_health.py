from graph_layout_rag.eval import corpus_health


def test_audit_chunk_distribution_reports_expected_counts(monkeypatch):
    rows = [
        {"doc_id": "paper-a", "text": "a", "canonical_sha256": "sha-a"},
        {"doc_id": "paper-a", "text": "b", "canonical_sha256": "sha-a"},
        {"doc_id": "paper-b", "text": "c", "canonical_sha256": "sha-b"},
        {"doc_id": "paper-b", "text": "d", "canonical_sha256": "sha-b"},
        {"doc_id": "paper-b", "text": "e", "canonical_sha256": "sha-b"},
        {"doc_id": "paper-b", "text": "f", "canonical_sha256": "sha-b"},
        {"doc_id": "paper-b", "text": "g", "canonical_sha256": "sha-b"},
    ]
    monkeypatch.setattr(corpus_health, "_load_chunks", lambda profile: rows)
    monkeypatch.setattr(corpus_health, "_pdfs_on_disk", lambda: {"paper-a", "paper-b"})

    findings = corpus_health.audit_chunk_distribution("test-profile", min_mean=2.0)

    summary = next(f for f in findings if f.code == "chunk_summary")
    assert summary.detail["chunks"] == 7
    assert summary.detail["docs"] == 2
    assert summary.detail["pdfs_on_disk"] == 2
    assert any(f.code == "chunk_density_ok" for f in findings)


def test_run_audit_sorts_critical_first(monkeypatch):
    monkeypatch.setattr(
        corpus_health,
        "audit_chunk_distribution",
        lambda profile: [corpus_health.Finding("critical", "no_index", "missing")],
    )
    monkeypatch.setattr(corpus_health, "audit_fulltext_stratification", lambda profile: [])
    monkeypatch.setattr(corpus_health, "audit_extraction_fallback", lambda: [])
    monkeypatch.setattr(corpus_health, "audit_credentials", lambda: [])
    monkeypatch.setattr(corpus_health, "audit_pool_holes", lambda track: [])

    report = corpus_health.run_audit("missing-profile")

    assert report["worst_severity"] == "critical"
    assert report["n_critical"] == 1
    assert report["findings"][0]["code"] == "no_index"


def test_fulltext_stratification_flags_metadata_ceiling(monkeypatch):
    from types import SimpleNamespace

    rows = [
        {"doc_id": "ft-a"},
        {"doc_id": "ft-a"},
        {"doc_id": "ft-a"},  # full-text doc, 3 chunks
        {"doc_id": "meta-b"},  # metadata-only, 1 chunk
        {"doc_id": "meta-c"},  # metadata-only, 1 chunk
        {"doc_id": "ghost-d"},  # unkeyed: not present in the manifest
    ]
    by_id = {
        "ft-a": SimpleNamespace(status="ok", localPath="data/raw/pdf/ft-a.pdf"),
        "meta-b": SimpleNamespace(status="metadata_only", localPath=None),
        "meta-c": SimpleNamespace(status="metadata_only", localPath=None),
    }
    monkeypatch.setattr(corpus_health, "_load_chunks", lambda profile: rows)
    monkeypatch.setattr(corpus_health, "_manifest_by_id", lambda: by_id)

    findings = corpus_health.audit_fulltext_stratification("p")

    summary = next(f for f in findings if f.code == "fulltext_stratification")
    assert summary.detail["docs_by_stratum"] == {"full_text": 1, "metadata_only": 2, "unkeyed": 1}
    assert summary.detail["chunks_by_stratum"]["full_text"] == 3
    assert summary.detail["metadata_only_doc_fraction"] == 0.5  # 2 of 4 docs
    assert any(f.code == "metadata_only_ceiling" and f.severity == "warning" for f in findings)


def test_fulltext_stratification_quiet_when_mostly_fulltext(monkeypatch):
    from types import SimpleNamespace

    rows = [{"doc_id": f"ft-{i}"} for i in range(9)] + [{"doc_id": "meta-x"}]
    by_id = {f"ft-{i}": SimpleNamespace(status="ok", localPath=f"p/ft-{i}.pdf") for i in range(9)}
    by_id["meta-x"] = SimpleNamespace(status="metadata_only", localPath=None)
    monkeypatch.setattr(corpus_health, "_load_chunks", lambda profile: rows)
    monkeypatch.setattr(corpus_health, "_manifest_by_id", lambda: by_id)

    findings = corpus_health.audit_fulltext_stratification("p")

    assert not any(
        f.code == "metadata_only_ceiling" for f in findings
    )  # 10% < 40%, no ceiling flag


def test_fulltext_stratification_empty_index_is_silent(monkeypatch):
    monkeypatch.setattr(corpus_health, "_load_chunks", lambda profile: [])
    assert corpus_health.audit_fulltext_stratification("p") == []
