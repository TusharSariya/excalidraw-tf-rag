"""Resumable metadata-enrichment backfill (all network mocked).

Covers the OpenAlex / S2 / arXiv extractors, the DOI-absent fallback path, the
interrupt-safe resume ordering, and that --dry-run writes nothing.
"""

from __future__ import annotations

import pytest

from graph_layout_rag import citation_store as cs
from graph_layout_rag import manifest as man_mod
from graph_layout_rag.harvest import enrich_metadata as em
from graph_layout_rag.harvest.providers import OutcomeKind, RequestOutcome
from graph_layout_rag.manifest import Manifest, ManifestItem, load_manifest, save_manifest


# --------------------------------------------------------------------------- fixtures

def _item(item_id: str, **kw) -> ManifestItem:
    base = dict(
        id=item_id,
        title=f"Title {item_id}",
        source="openalex",
        url=f"https://example.org/{item_id}",
        status="ok",
    )
    base.update(kw)
    return ManifestItem(**base)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Redirect both the citations DB and the manifest at temp paths."""
    db_path = tmp_path / "citations.sqlite"
    manifest_path = tmp_path / "manifest.json"
    monkeypatch.setattr("graph_layout_rag.citation_store.CITATIONS_DB_PATH", db_path)
    monkeypatch.setattr(man_mod, "MANIFEST_PATH", manifest_path)

    def write_manifest(items: list[ManifestItem]) -> None:
        save_manifest(Manifest(items=items))

    return type("Env", (), {
        "db_path": db_path,
        "manifest_path": manifest_path,
        "write": staticmethod(write_manifest),
    })


def _ok(data):
    return RequestOutcome(OutcomeKind.SUCCESS, data=data, status_code=200)


def _miss():
    return RequestOutcome(OutcomeKind.TERMINAL_MISS, status_code=404)


def _rate_limited():
    return RequestOutcome(OutcomeKind.RATE_LIMITED, status_code=429, retry_after=1)


# canned OpenAlex work
_FULL_WORK = {
    "id": "https://openalex.org/W1",
    "fwci": 3.5,
    "cited_by_count": 42,
    "is_retracted": False,
    "abstract_inverted_index": {"Graph": [0], "layout": [1], "is": [2], "hard": [3]},
    "primary_location": {
        "version": "publishedVersion",
        "source": {"display_name": "Graph Drawing", "type": "conference"},
    },
    "best_oa_location": {"pdf_url": "https://oa.example/p.pdf", "license": "cc-by"},
    "type": "article",
    "biblio": {"volume": "7", "issue": "2", "first_page": "10", "last_page": "20"},
    "authorships": [
        {"author": {"display_name": "Ada Lovelace"}},
        {"author": {"display_name": "Alan Turing"}},
    ],
}


# --------------------------------------------------------------------------- extractor units

def test_openalex_extractor_full():
    out = em._extract_openalex(_FULL_WORK)
    assert out["fwci"] == 3.5
    assert out["cited_by_count"] == 42
    assert out["venue"] == "Graph Drawing"
    assert out["venue_type"] == "conference"
    assert out["oa_version"] == "publishedVersion"
    assert out["genre"] == "article"
    assert out["oa_pdf_url"] == "https://oa.example/p.pdf"
    assert out["license"] == "cc-by"
    assert out["biblio"]["volume"] == "7"
    assert out["full_authors"] == ["Ada Lovelace", "Alan Turing"]
    assert out["abstract"] == "Graph layout is hard"


def test_openalex_extractor_missing_fields():
    out = em._extract_openalex({"id": "https://openalex.org/W9"})
    assert out["fwci"] is None
    assert out["venue"] is None
    assert out["genre"] is None
    assert out["biblio"] is None
    assert out["full_authors"] is None
    assert out["abstract"] is None  # inverted index absent


def test_openalex_inverted_index_none():
    out = em._extract_openalex({"abstract_inverted_index": None})
    assert out["abstract"] is None


def test_inverted_index_word_order():
    # words deliberately out of order in the dict; positions define the sentence
    idx = {"world": [1], "hello": [0], "again": [3], "hello,": [2]}
    out = em._extract_openalex({"abstract_inverted_index": idx})
    assert out["abstract"] == "hello world hello, again"


# --------------------------------------------------------------------------- S2 batch

def test_s2_batch_partial_and_specter_roundtrip(env, monkeypatch):
    env.write([_item("a", doi="10.1/aaa"), _item("b", doi="10.1/bbb")])

    # OpenAlex misses both so the only data comes from S2.
    monkeypatch.setattr(em.OPENALEX, "request", lambda *a, **k: _miss())
    monkeypatch.setattr(em, "_arxiv_query", lambda ids: None)

    def s2_request(method, url, **kw):
        # one real paper (with specter vector), one null
        return _ok([
            {"tldr": {"text": "a tldr"}, "publicationTypes": ["JournalArticle"],
             "openAccessPdf": {"url": "https://s2.example/a.pdf"},
             "embedding": {"model": "specter_v2", "vector": [0.1, 0.2, 0.3]},
             "externalIds": {}},
            None,
        ])

    monkeypatch.setattr(em.SEMANTIC_SCHOLAR, "request", s2_request)

    stats = em.enrich_metadata(workers=1)
    assert stats["enriched"] == 1          # item a
    assert stats["terminal_miss"] == 1     # item b (s2 null, oa miss, no arxiv)
    assert stats["specter2_stored"] == 1

    db = cs.connect(env.db_path)
    meta = cs.paper_meta_for_doc(db, "a")
    assert meta.tldr == "a tldr"
    assert meta.oa_pdf_url == "https://s2.example/a.pdf"
    assert meta.source_provider == "s2"
    assert cs.specter2_for_doc(db, "a") == pytest.approx([0.1, 0.2, 0.3], abs=1e-6)
    # terminal-miss marker present so it is not retried
    assert cs.paper_meta_for_doc(db, "b").source_provider == "none"


def test_s2_transient_defers_no_row(env, monkeypatch):
    env.write([_item("a", doi="10.1/aaa")])
    monkeypatch.setattr(em.OPENALEX, "request", lambda *a, **k: _miss())
    monkeypatch.setattr(em, "_arxiv_query", lambda ids: None)
    monkeypatch.setattr(em.SEMANTIC_SCHOLAR, "request", lambda *a, **k: _rate_limited())

    stats = em.enrich_metadata(workers=1)
    assert stats["transient_deferred"] == 1
    assert stats["enriched"] == 0
    assert stats["terminal_miss"] == 0

    db = cs.connect(env.db_path)
    assert not cs.has_paper_meta(db, "a")  # deferred -> no row


# --------------------------------------------------------------------------- fallback path

def test_doi_absent_uses_arxiv_and_title_fallback(env, monkeypatch):
    # item has NO doi but an arXiv id in externalIds
    env.write([_item("a", externalIds={"ArXiv": "2401.00001"})])

    seen_lookups: list = []

    def oa_request(method, url, *, params=None, **kw):
        seen_lookups.append((url, params))
        # arxiv-DOI form misses; the title-search fallback resolves
        if (params or {}).get("search"):
            return _ok({"results": [dict(_FULL_WORK)]})
        return _miss()

    monkeypatch.setattr(em.OPENALEX, "request", oa_request)
    monkeypatch.setattr(em.SEMANTIC_SCHOLAR, "request", lambda *a, **k: _ok([None]))
    monkeypatch.setattr(em, "_arxiv_query", lambda ids: None)

    stats = em.enrich_metadata(workers=1)
    assert stats["enriched"] == 1
    # both the arxiv-DOI form and the title-search fallback must have been tried
    joined = " ".join(u for u, _ in seen_lookups).lower()
    assert "arxiv" in joined
    assert any((p or {}).get("search") for _, p in seen_lookups)


def test_arxiv_extractor_categories(env, monkeypatch):
    env.write([_item("a", externalIds={"ArXiv": "2401.00002"})])
    monkeypatch.setattr(em.OPENALEX, "request", lambda *a, **k: _miss())
    monkeypatch.setattr(em.SEMANTIC_SCHOLAR, "request", lambda *a, **k: _ok([None]))

    xml = (
        '<feed xmlns="http://www.w3.org/2005/Atom" '
        'xmlns:arxiv="http://arxiv.org/schemas/atom">'
        '<entry>'
        '<id>http://arxiv.org/abs/2401.00002v1</id>'
        '<arxiv:primary_category term="cs.CG"/>'
        '<category term="cs.CG"/>'
        '<category term="cs.DS"/>'
        '</entry></feed>'
    )
    monkeypatch.setattr(em, "_arxiv_query", lambda ids: xml)

    stats = em.enrich_metadata(workers=1)
    assert stats["enriched"] == 1

    m = load_manifest()
    item = {i.id: i for i in m.items}["a"]
    assert item.arxiv_category == "cs.CG,cs.DS"
    assert item.externalIds.get("ArXiv") == "2401.00002"


# --------------------------------------------------------------------------- manifest mapping

def test_manifest_fields_and_abstract_backfill(env, monkeypatch):
    env.write([_item("a", doi="10.1/aaa")])  # no abstract initially
    monkeypatch.setattr(em.OPENALEX, "request", lambda *a, **k: _ok(dict(_FULL_WORK)))
    monkeypatch.setattr(em.SEMANTIC_SCHOLAR, "request", lambda *a, **k: _ok([None]))
    monkeypatch.setattr(em, "_arxiv_query", lambda ids: None)

    stats = em.enrich_metadata(workers=1)
    assert stats["abstracts_backfilled"] == 1

    item = {i.id: i for i in load_manifest().items}["a"]
    assert item.venue == "Graph Drawing"
    assert item.venue_type == "conference"
    assert item.oa_version == "publishedVersion"
    assert item.genre == "article"
    assert item.abstract == "Graph layout is hard"

    db = cs.connect(env.db_path)
    meta = cs.paper_meta_for_doc(db, "a")
    assert meta.fwci == 3.5
    assert meta.cited_by_count == 42
    assert meta.full_authors == ["Ada Lovelace", "Alan Turing"]
    assert meta.source_provider == "openalex"


# --------------------------------------------------------------------------- resumability

def test_resume_skips_done_and_marker_not_refetched(env, monkeypatch):
    env.write([
        _item("done", doi="10.1/done"),      # gets data -> enriched
        _item("term", doi="10.1/term"),      # clean miss -> none marker
        _item("flaky", doi="10.1/flaky"),    # transient -> deferred (no row)
    ])

    # First run: done resolves on OpenAlex; term clean-misses; flaky is rate-limited.
    def oa_run1(method, url, *, params=None, **kw):
        if "done" in url:
            return _ok(dict(_FULL_WORK))
        if "flaky" in url:
            return _rate_limited()
        return _miss()

    monkeypatch.setattr(em.OPENALEX, "request", oa_run1)
    # S2 cleanly misses all three (null papers); the only transient signal is OpenAlex/flaky.
    monkeypatch.setattr(em.SEMANTIC_SCHOLAR, "request", lambda *a, **k: _ok([None, None, None]))
    monkeypatch.setattr(em, "_arxiv_query", lambda ids: None)

    s1 = em.enrich_metadata(workers=1)
    assert s1["enriched"] == 1
    assert s1["terminal_miss"] == 1
    assert s1["transient_deferred"] == 1

    db = cs.connect(env.db_path)
    assert cs.has_paper_meta(db, "done")
    assert cs.has_paper_meta(db, "term")       # marker row
    assert not cs.has_paper_meta(db, "flaky")  # deferred
    db.close()

    # Second run: only "flaky" should be re-fetched. Assert done/term are NOT requested.
    requested: list[str] = []

    def oa_run2(method, url, *, params=None, **kw):
        requested.append(url)
        if "flaky" in url:
            return _ok(dict(_FULL_WORK))
        return _miss()

    monkeypatch.setattr(em.OPENALEX, "request", oa_run2)
    monkeypatch.setattr(em.SEMANTIC_SCHOLAR, "request", lambda *a, **k: _ok([None]))

    s2 = em.enrich_metadata(workers=1)
    assert s2["scanned"] == 1            # only the unfinished item
    assert s2["enriched"] == 1
    joined = " ".join(requested)
    assert "flaky" in joined
    assert "done" not in joined          # already-done not re-fetched
    assert "term" not in joined          # terminal-miss marker not re-fetched

    db = cs.connect(env.db_path)
    assert cs.has_paper_meta(db, "flaky")


# --------------------------------------------------------------------------- dry-run

def test_dry_run_writes_nothing(env, monkeypatch):
    env.write([_item("a", doi="10.1/aaa")])
    before = env.manifest_path.read_text(encoding="utf-8")

    monkeypatch.setattr(em.OPENALEX, "request", lambda *a, **k: _ok(dict(_FULL_WORK)))
    monkeypatch.setattr(em.SEMANTIC_SCHOLAR, "request", lambda *a, **k: _ok([None]))
    monkeypatch.setattr(em, "_arxiv_query", lambda ids: None)

    stats = em.enrich_metadata(workers=1, dry_run=True)
    assert stats["enriched"] == 1  # counts computed

    # manifest untouched, no papers_meta row
    assert env.manifest_path.read_text(encoding="utf-8") == before
    db = cs.connect(env.db_path)
    assert not cs.has_paper_meta(db, "a")
