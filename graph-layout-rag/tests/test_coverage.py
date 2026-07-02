from types import SimpleNamespace

from graph_layout_rag.eval import coverage


class _Case:
    def __init__(self, id, relevant):
        self.id = id
        self.relevant_doc_ids = frozenset(relevant)


def test_compute_coverage_decomposes_recall_ceiling(monkeypatch):
    cases = [
        _Case("q1", {"ft-a", "meta-b", "missing-x"}),  # 2 of 3 recoverable
        _Case("q2", {"ft-a"}),  # fully recoverable
    ]
    indexed = {"ft-a", "meta-b"}  # missing-x not in the index
    by_id = {
        "ft-a": SimpleNamespace(status="ok", localPath="p/ft-a.pdf"),
        "meta-b": SimpleNamespace(status="metadata_only", localPath=None),
    }
    monkeypatch.setattr(coverage, "cases_for_track", lambda track: cases)
    monkeypatch.setattr(
        coverage.corpus_health, "_load_chunks", lambda profile: [{"doc_id": d} for d in indexed]
    )
    monkeypatch.setattr(coverage.corpus_health, "_manifest_by_id", lambda: by_id)

    out = coverage.compute_coverage("p", track="catalog")

    assert out["total_relevant"] == 4  # 3 + 1
    assert out["relevant_by_stratum"] == {"full_text": 2, "metadata_only": 1, "not_indexed": 1}
    # 3 of 4 relevant docs are recoverable
    assert out["recall_ceiling"] == 0.75
    q1 = next(c for c in out["per_case"] if c["id"] == "q1")
    assert q1["recall_ceiling"] == round(2 / 3, 3)
    assert q1["missing"] == ["missing-x"]


def test_compute_coverage_empty_gold(monkeypatch):
    monkeypatch.setattr(coverage, "cases_for_track", lambda track: [])
    monkeypatch.setattr(coverage.corpus_health, "_load_chunks", lambda profile: [])
    monkeypatch.setattr(coverage.corpus_health, "_manifest_by_id", lambda: {})
    out = coverage.compute_coverage("p")
    assert out["recall_ceiling"] == 0.0
    assert out["total_relevant"] == 0
