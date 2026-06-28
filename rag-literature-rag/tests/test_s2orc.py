from rag_literature_rag.harvest import s2orc
from rag_literature_rag.harvest.providers import OutcomeKind, RequestOutcome
from rag_literature_rag.harvest.s2orc import (
    RAG_TOPIC_QUERIES,
    _authors,
    _paper_to_spec,
    _pdf_url,
    harvest_s2orc,
)


def _relevant_paper():
    return {
        "title": "Dense Passage Retrieval for Open-Domain Question Answering",
        "year": 2020,
        "abstract": "Dense passage retrieval with retrieval augmented generation.",
        "externalIds": {"DOI": "10.1/DPR-1"},
        "authors": [{"name": "Vladimir Karpukhin"}],
        "openAccessPdf": {"url": "https://s2.example/dpr.pdf", "status": "GREEN"},
    }


def _offtopic_paper():
    return {
        "title": "Wind Turbine Blade Fatigue Analysis",
        "year": 2018,
        "abstract": "Structural analysis of wind turbine blades.",
        "externalIds": {"DOI": "10.1/wt-1"},
        "authors": [{"name": "Someone"}],
        "openAccessPdf": {"url": "https://s2.example/wt.pdf"},
    }


def test_topic_queries_present():
    assert "agentic rag" in RAG_TOPIC_QUERIES
    assert len(RAG_TOPIC_QUERIES) >= 10


def test_field_parsers():
    p = _relevant_paper()
    assert _authors(p) == ["Vladimir Karpukhin"]
    assert _pdf_url(p) == "https://s2.example/dpr.pdf"
    spec = _paper_to_spec(p, tags=["dense-retrieval"])
    assert spec["doi"] == "10.1/dpr-1"
    assert spec["year"] == 2020


def test_paper_to_spec_requires_title():
    assert _paper_to_spec({"externalIds": {"DOI": "10.1/x"}}, tags=[]) is None


def test_harvest_s2orc_shape_and_gating(monkeypatch):
    def fake_request(method, url, **kwargs):
        params = kwargs["params"]
        if "token" not in params:
            return RequestOutcome(
                OutcomeKind.SUCCESS,
                data={"data": [_relevant_paper(), _offtopic_paper()], "token": None},
                status_code=200,
            )
        return RequestOutcome(OutcomeKind.SUCCESS, data={"data": []}, status_code=200)

    monkeypatch.setattr(s2orc.SEMANTIC_SCHOLAR, "request", fake_request)
    monkeypatch.setattr(
        s2orc, "download_to_file", lambda *a, **k: {"ok": False, "sha256": None}
    )

    items = harvest_s2orc(max_works=50, workers=1)

    assert isinstance(items, list) and items
    assert all(it.source == "s2orc" for it in items)
    assert all("semantic-scholar" in it.tags for it in items)
    titles = {it.title for it in items}
    assert any("Dense Passage Retrieval" in t for t in titles)
    assert not any("Wind Turbine" in t for t in titles)
    assert len({it.doi for it in items}) == 1


def test_harvest_s2orc_respects_existing_ids(monkeypatch):
    def fake_request(method, url, **kwargs):
        if "token" not in kwargs["params"]:
            return RequestOutcome(
                OutcomeKind.SUCCESS, data={"data": [_relevant_paper()], "token": None}, status_code=200
            )
        return RequestOutcome(OutcomeKind.SUCCESS, data={"data": []}, status_code=200)

    monkeypatch.setattr(s2orc.SEMANTIC_SCHOLAR, "request", fake_request)
    monkeypatch.setattr(
        s2orc, "download_to_file", lambda *a, **k: {"ok": False, "sha256": None}
    )

    from rag_literature_rag.manifest import slug_id

    existing = {slug_id("s2orc-10.1/dpr-1")}
    items = harvest_s2orc(max_works=50, workers=1, existing_ids=existing)
    assert items == []


def test_harvest_s2orc_download_ok(monkeypatch, tmp_path):
    monkeypatch.setattr(s2orc, "PDF_DIR", tmp_path)
    monkeypatch.setattr(s2orc, "relative_local_path", lambda p: f"data/raw/pdf/{p.name}")

    def fake_request(method, url, **kwargs):
        if "token" not in kwargs["params"]:
            return RequestOutcome(
                OutcomeKind.SUCCESS, data={"data": [_relevant_paper()], "token": None}, status_code=200
            )
        return RequestOutcome(OutcomeKind.SUCCESS, data={"data": []}, status_code=200)

    monkeypatch.setattr(s2orc.SEMANTIC_SCHOLAR, "request", fake_request)

    def fake_download(dest, url, **kwargs):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"%PDF-1.7 fake")
        return {"ok": True, "sha256": "cafe"}

    monkeypatch.setattr(s2orc, "download_to_file", fake_download)

    items = harvest_s2orc(max_works=10, workers=1)
    assert len(items) == 1
    assert items[0].status == "ok"
    assert items[0].sha256 == "cafe"
