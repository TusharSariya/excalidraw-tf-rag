from rag_literature_rag.harvest import core_source
from rag_literature_rag.harvest.core_source import (
    RAG_TOPIC_QUERIES,
    _authors,
    _pdf_urls,
    _result_to_spec,
    _year,
    harvest_core,
)
from rag_literature_rag.harvest.providers import OutcomeKind, RequestOutcome


def _relevant_result():
    return {
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive Tasks",
        "doi": "10.1/RAG-A",
        "authors": [{"name": "Patrick Lewis"}],
        "yearPublished": 2020,
        "abstract": "We study retrieval augmented generation.",
        "downloadUrl": "https://core.ac.uk/x.pdf",
        "sourceFulltextUrls": ["https://repo.example/x.pdf"],
    }


def _offtopic_result():
    return {
        "title": "A Study of Protein Folding Kinetics in Yeast",
        "doi": "10.1/bio-1",
        "authors": [{"name": "Someone"}],
        "yearPublished": 2019,
        "abstract": "Genome sequencing and folding.",
        "downloadUrl": "https://core.ac.uk/bio.pdf",
    }


def test_topic_queries_present():
    assert "graphrag" in RAG_TOPIC_QUERIES
    assert len(RAG_TOPIC_QUERIES) >= 10


def test_field_parsers():
    r = _relevant_result()
    assert _year(r) == 2020
    assert _authors(r) == ["Patrick Lewis"]
    assert _pdf_urls(r)[0] == "https://core.ac.uk/x.pdf"


def test_result_to_spec_requires_title():
    assert _result_to_spec({"doi": "10.1/x"}, tags=[]) is None


def test_harvest_core_shape_and_gating(monkeypatch):
    def fake_request(method, url, **kwargs):
        # one page per topic, then stop on empty
        offset = kwargs["params"]["offset"]
        if offset == "0":
            data = {"results": [_relevant_result(), _offtopic_result()]}
        else:
            data = {"results": []}
        return RequestOutcome(OutcomeKind.SUCCESS, data=data, status_code=200)

    monkeypatch.setattr(core_source.CORE, "request", fake_request)
    monkeypatch.setattr(
        core_source, "download_to_file", lambda *a, **k: {"ok": False, "sha256": None}
    )

    items = harvest_core(max_works=50, workers=1)

    assert isinstance(items, list) and items
    # off-topic protein-folding result is gated out; only the RAG one survives (deduped by DOI)
    assert all(it.source == "core" for it in items)
    assert all("core" in it.tags for it in items)
    titles = {it.title for it in items}
    assert any("Retrieval-Augmented Generation" in t for t in titles)
    assert not any("Protein Folding" in t for t in titles)
    # exactly one unique DOI kept
    assert len({it.doi for it in items}) == 1


def test_harvest_core_respects_existing_ids(monkeypatch):
    def fake_request(method, url, **kwargs):
        if kwargs["params"]["offset"] == "0":
            return RequestOutcome(
                OutcomeKind.SUCCESS, data={"results": [_relevant_result()]}, status_code=200
            )
        return RequestOutcome(OutcomeKind.SUCCESS, data={"results": []}, status_code=200)

    monkeypatch.setattr(core_source.CORE, "request", fake_request)
    monkeypatch.setattr(
        core_source, "download_to_file", lambda *a, **k: {"ok": False, "sha256": None}
    )

    from rag_literature_rag.manifest import slug_id

    existing = {slug_id("core-10.1/rag-a")}
    items = harvest_core(max_works=50, workers=1, existing_ids=existing)
    assert items == []


def test_harvest_core_download_ok(monkeypatch, tmp_path):
    monkeypatch.setattr(core_source, "PDF_DIR", tmp_path)
    monkeypatch.setattr(core_source, "relative_local_path", lambda p: f"data/raw/pdf/{p.name}")

    def fake_request(method, url, **kwargs):
        if kwargs["params"]["offset"] == "0":
            return RequestOutcome(
                OutcomeKind.SUCCESS, data={"results": [_relevant_result()]}, status_code=200
            )
        return RequestOutcome(OutcomeKind.SUCCESS, data={"results": []}, status_code=200)

    monkeypatch.setattr(core_source.CORE, "request", fake_request)

    def fake_download(dest, url, **kwargs):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"%PDF-1.7 fake")
        return {"ok": True, "sha256": "deadbeef"}

    monkeypatch.setattr(core_source, "download_to_file", fake_download)

    items = harvest_core(max_works=10, workers=1)
    assert len(items) == 1
    assert items[0].status == "ok"
    assert items[0].sha256 == "deadbeef"
    assert items[0].contentType == "application/pdf"
