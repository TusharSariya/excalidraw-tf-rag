from unittest.mock import MagicMock

from rag_literature_rag.harvest import doab
from rag_literature_rag.harvest.doab import (
    RAG_TOPIC_QUERIES,
    _bitstream_pdf_url,
    _item_to_spec,
    _metadata_map,
    harvest_doab,
)


def _relevant_item():
    return {
        "name": "Retrieval-Augmented Generation: An Open Access Handbook",
        "handle": "20.500/1",
        "uuid": "uuid-rag",
        "metadata": [
            {"key": "dc.contributor.author", "value": "Ada Lovelace"},
            {"key": "dc.date.issued", "value": "2024-03-01"},
            {"key": "dc.description.abstract", "value": "A book on retrieval augmented generation."},
            {"key": "dc.identifier", "value": "10.1/doab-rag"},
        ],
        "bitstreams": [
            {
                "bundleName": "ORIGINAL",
                "mimeType": "application/pdf",
                "retrieveLink": "/bitstreams/1/retrieve",
                "metadata": [],
            }
        ],
    }


def _offtopic_item():
    return {
        "name": "Petroleum Drilling Operations Handbook",
        "handle": "20.500/2",
        "uuid": "uuid-oil",
        "metadata": [
            {"key": "dc.contributor.author", "value": "Someone"},
            {"key": "dc.date.issued", "value": "2019"},
            {"key": "dc.description.abstract", "value": "Drilling operations and reservoir simulation."},
        ],
        "bitstreams": [
            {
                "bundleName": "ORIGINAL",
                "mimeType": "application/pdf",
                "retrieveLink": "/bitstreams/2/retrieve",
            }
        ],
    }


def _mock_client(pages):
    """pages: list of JSON payloads returned in order from client.get()."""
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.get.side_effect = [
        MagicMock(status_code=200, json=MagicMock(return_value=p)) for p in pages
    ]
    return client


def test_topic_queries_present():
    assert "chunking retrieval" in RAG_TOPIC_QUERIES
    assert len(RAG_TOPIC_QUERIES) >= 10


def test_metadata_and_bitstream_parsers():
    item = _relevant_item()
    md = _metadata_map(item)
    assert md["dc.contributor.author"] == ["Ada Lovelace"]
    assert _bitstream_pdf_url(item) == "https://directory.doabooks.org/bitstreams/1/retrieve"


def test_bitstream_prefers_oapen_download_url():
    item = {
        "bitstreams": [
            {
                "bundleName": "ORIGINAL",
                "mimeType": "application/pdf",
                "retrieveLink": "/x",
                "metadata": [{"key": "oapen.identifier.downloadUrl", "value": "https://oapen/x.pdf"}],
            }
        ]
    }
    assert _bitstream_pdf_url(item) == "https://oapen/x.pdf"


def test_item_to_spec_requires_title():
    assert _item_to_spec({"handle": "x"}, tags=[]) is None


def test_item_to_spec_fields():
    spec = _item_to_spec(_relevant_item(), tags=["foundations"])
    assert spec["title"].startswith("Retrieval-Augmented Generation")
    assert spec["authors"] == ["Ada Lovelace"]
    assert spec["year"] == 2024
    assert spec["doi"] == "10.1/doab-rag"


def test_harvest_doab_shape_and_gating(monkeypatch):
    # Each topic loop: offset 0 -> one page with both items, then empty.
    def client_factory(*a, **k):
        return _mock_client([[_relevant_item(), _offtopic_item()], []])

    monkeypatch.setattr(doab.httpx, "Client", client_factory)
    monkeypatch.setattr(doab, "download_to_file", lambda *a, **k: {"ok": False, "sha256": None})

    items = harvest_doab(max_works=50, workers=1)

    assert isinstance(items, list) and items
    assert all(it.source == "doab" for it in items)
    assert all("book" in it.tags and "open-access-book" in it.tags for it in items)
    titles = {it.title for it in items}
    assert any("Retrieval-Augmented Generation" in t for t in titles)
    assert not any("Petroleum" in t for t in titles)
    assert len({it.doi for it in items}) == 1


def test_harvest_doab_respects_existing_ids(monkeypatch):
    def client_factory(*a, **k):
        return _mock_client([[_relevant_item()], []])

    monkeypatch.setattr(doab.httpx, "Client", client_factory)
    monkeypatch.setattr(doab, "download_to_file", lambda *a, **k: {"ok": False, "sha256": None})

    from rag_literature_rag.manifest import slug_id

    existing = {slug_id("doab-10.1/doab-rag")}
    items = harvest_doab(max_works=50, workers=1, existing_ids=existing)
    assert items == []


def test_harvest_doab_download_ok(monkeypatch, tmp_path):
    monkeypatch.setattr(doab, "PDF_DIR", tmp_path)
    monkeypatch.setattr(doab, "relative_local_path", lambda p: f"data/raw/pdf/{p.name}")

    def client_factory(*a, **k):
        return _mock_client([[_relevant_item()], []])

    monkeypatch.setattr(doab.httpx, "Client", client_factory)

    def fake_download(dest, url, **kwargs):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"%PDF-1.7 fake")
        return {"ok": True, "sha256": "b00c"}

    monkeypatch.setattr(doab, "download_to_file", fake_download)

    items = harvest_doab(max_works=10, workers=1)
    assert len(items) == 1
    assert items[0].status == "ok"
    assert items[0].sha256 == "b00c"
