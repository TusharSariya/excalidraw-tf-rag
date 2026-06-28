from rag_literature_rag.harvest import pmc
from rag_literature_rag.harvest.pmc import (
    RAG_TOPIC_QUERIES,
    _authors,
    _pdf_url,
    _result_to_spec,
    harvest_pmc,
)


def _relevant_result():
    return {
        "title": "Retrieval-Augmented Generation for Biomedical Question Answering",
        "doi": "10.1/RAG-BIO",
        "authorString": "Smith J, Doe A",
        "pubYear": "2023",
        "abstractText": "We apply retrieval augmented generation to QA.",
        "pmcid": "PMC123",
        "fullTextUrlList": {
            "fullTextUrl": [
                {"availability": "Open access", "documentStyle": "html", "url": "https://x/html"},
                {"availability": "Open access", "documentStyle": "pdf", "url": "https://x/pdf"},
            ]
        },
    }


def _offtopic_result():
    return {
        "title": "Oncology Clinical Trial Outcomes in Stage IV Patients",
        "doi": "10.1/onc-1",
        "authorString": "Person X",
        "pubYear": "2021",
        "abstractText": "Clinical trial and oncology results.",
        "pmcid": "PMC999",
        "fullTextUrlList": {
            "fullTextUrl": [
                {"availability": "Open access", "documentStyle": "pdf", "url": "https://x/onc.pdf"}
            ]
        },
    }


def test_topic_queries_present():
    assert "rag evaluation" in RAG_TOPIC_QUERIES
    assert len(RAG_TOPIC_QUERIES) >= 10


def test_field_parsers():
    r = _relevant_result()
    assert _authors(r) == ["Smith J", "Doe A"]
    assert _pdf_url(r) == "https://x/pdf"
    spec = _result_to_spec(r, tags=["foundations"])
    assert spec["doi"] == "10.1/rag-bio"
    assert spec["year"] == 2023


def test_authors_from_author_list():
    r = {"authorList": {"author": [{"fullName": "Jane Roe"}]}}
    assert _authors(r) == ["Jane Roe"]


def test_result_to_spec_requires_title():
    assert _result_to_spec({"doi": "10.1/x"}, tags=[]) is None


def test_harvest_pmc_shape_and_gating(monkeypatch):
    calls = {"n": 0}

    def fake_get_json(url, **kwargs):
        calls["n"] += 1
        cursor = kwargs["params"]["cursorMark"]
        if cursor == "*":
            return {
                "resultList": {"result": [_relevant_result(), _offtopic_result()]},
                "nextCursorMark": "next",
            }
        return {"resultList": {"result": []}, "nextCursorMark": "next"}

    monkeypatch.setattr(pmc, "get_json", fake_get_json)
    monkeypatch.setattr(pmc, "download_to_file", lambda *a, **k: {"ok": False, "sha256": None})

    items = harvest_pmc(max_works=50, workers=1)

    assert isinstance(items, list) and items
    assert all(it.source == "europepmc" for it in items)
    assert all("pmc" in it.tags and "biomedical" in it.tags for it in items)
    titles = {it.title for it in items}
    assert any("Retrieval-Augmented Generation" in t for t in titles)
    assert not any("Oncology" in t for t in titles)
    assert len({it.doi for it in items}) == 1


def test_harvest_pmc_respects_existing_ids(monkeypatch):
    def fake_get_json(url, **kwargs):
        if kwargs["params"]["cursorMark"] == "*":
            return {"resultList": {"result": [_relevant_result()]}, "nextCursorMark": "next"}
        return {"resultList": {"result": []}, "nextCursorMark": "next"}

    monkeypatch.setattr(pmc, "get_json", fake_get_json)
    monkeypatch.setattr(pmc, "download_to_file", lambda *a, **k: {"ok": False, "sha256": None})

    from rag_literature_rag.manifest import slug_id

    existing = {slug_id("europepmc-10.1/rag-bio")}
    items = harvest_pmc(max_works=50, workers=1, existing_ids=existing)
    assert items == []


def test_harvest_pmc_download_ok(monkeypatch, tmp_path):
    monkeypatch.setattr(pmc, "PDF_DIR", tmp_path)
    monkeypatch.setattr(pmc, "relative_local_path", lambda p: f"data/raw/pdf/{p.name}")

    def fake_get_json(url, **kwargs):
        if kwargs["params"]["cursorMark"] == "*":
            return {"resultList": {"result": [_relevant_result()]}, "nextCursorMark": "next"}
        return {"resultList": {"result": []}, "nextCursorMark": "next"}

    monkeypatch.setattr(pmc, "get_json", fake_get_json)

    def fake_download(dest, url, **kwargs):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"%PDF-1.7 fake")
        return {"ok": True, "sha256": "f00d"}

    monkeypatch.setattr(pmc, "download_to_file", fake_download)

    items = harvest_pmc(max_works=10, workers=1)
    assert len(items) == 1
    assert items[0].status == "ok"
    assert items[0].sha256 == "f00d"
