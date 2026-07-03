"""Offline unit tests for rag_common.citation_gaps against a hand-built fixture DB."""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

from rag_common.citation_gaps import (
    GapWork,
    compute_gaps,
    fetch_openalex_titles,
    resolve_gap_titles,
)


def _fixture_db() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript(
        """
        CREATE TABLE papers(
          oa_id TEXT PRIMARY KEY,
          doi TEXT,
          doc_id TEXT,
          title TEXT,
          year INTEGER,
          cited_by_count INTEGER DEFAULT 0,
          in_corpus INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE cites(
          src_oa TEXT NOT NULL,
          dst_oa TEXT NOT NULL,
          is_influential INTEGER NOT NULL DEFAULT 0,
          PRIMARY KEY(src_oa, dst_oa)
        );
        CREATE TABLE paper_aliases(
          provider TEXT NOT NULL,
          external_id TEXT NOT NULL,
          oa_id TEXT NOT NULL,
          PRIMARY KEY(provider, external_id)
        );
        """
    )
    # 3 corpus docs (W1, W2, W3, all in_corpus=1).
    db.executemany(
        "INSERT INTO papers(oa_id, doi, doc_id, title, year, in_corpus) VALUES(?,?,?,?,?,?)",
        [
            ("W1", "10.1/a", "doc-a", "Paper A", 2020, 1),
            ("W2", "10.1/b", "doc-b", "Paper B", 2021, 1),
            ("W3", "10.1/c", "doc-c", "Paper C", 2022, 1),
        ],
    )
    # External works: W10 cited by all 3 corpus docs (title known), W11 cited by 2
    # (no title -- degrade to DOI), W12 cited by 1 (below default min-citations),
    # W13 has only an arXiv alias and no title/doi, W14 is external but already
    # `in_corpus=0` with a row present (still a gap), W2 also cites W1 (in-corpus
    # target -- must never appear as a gap).
    db.executemany(
        "INSERT INTO papers(oa_id, doi, doc_id, title, year, in_corpus) VALUES(?,?,?,?,?,?)",
        [
            ("W10", "10.2/known", None, "Well-Known External Paper", 2015, 0),
            ("W11", "10.2/notitle", None, None, None, 0),
            ("W12", "10.2/onecite", None, "Rarely Cited External", 2010, 0),
            ("W13", None, None, None, None, 0),
        ],
    )
    db.executemany(
        "INSERT INTO cites(src_oa, dst_oa) VALUES(?,?)",
        [
            ("W1", "W10"),
            ("W2", "W10"),
            ("W3", "W10"),
            ("W1", "W11"),
            ("W2", "W11"),
            ("W1", "W12"),
            ("W1", "W13"),
            ("W2", "W13"),
            ("W2", "W1"),  # in-corpus target -- must be excluded from gaps
            # W14 has no papers row at all (never resolved) but is still a real gap.
            ("W1", "W14"),
            ("W2", "W14"),
        ],
    )
    db.executemany(
        "INSERT INTO paper_aliases(provider, external_id, oa_id) VALUES(?,?,?)",
        [("arxiv", "2001.00001", "W13")],
    )
    db.commit()
    return db


def test_compute_gaps_ranking_and_filter():
    db = _fixture_db()
    gaps = compute_gaps(db, min_citations=2, top=50)

    oa_ids = [g.oa_id for g in gaps]
    # W12 has only 1 in-corpus citer -- filtered by min_citations=2.
    assert "W12" not in oa_ids
    # In-corpus target W1 must never appear, even though W2 cites it.
    assert "W1" not in oa_ids

    # Ranked desc by distinct in-corpus citer count: W10 (3) > {W11, W13, W14} (2 each).
    assert gaps[0].oa_id == "W10"
    assert gaps[0].cited_by_count == 3
    for g in gaps[1:]:
        assert g.cited_by_count == 2


def test_compute_gaps_min_citations_default_excludes_singletons():
    db = _fixture_db()
    gaps = compute_gaps(db, top=50)  # default min_citations=2
    assert all(g.cited_by_count >= 2 for g in gaps)


def test_compute_gaps_top_caps_results():
    db = _fixture_db()
    gaps = compute_gaps(db, min_citations=1, top=1)
    assert len(gaps) == 1
    assert gaps[0].oa_id == "W10"


def test_compute_gaps_degrades_gracefully_without_title():
    db = _fixture_db()
    gaps = compute_gaps(db, min_citations=2, top=50)
    by_id = {g.oa_id: g for g in gaps}

    w11 = by_id["W11"]
    assert w11.title is None
    assert w11.doi == "10.2/notitle"
    assert w11.display_label() == "doi:10.2/notitle"

    w13 = by_id["W13"]
    assert w13.title is None
    assert w13.doi is None
    assert w13.arxiv_id == "2001.00001"
    assert w13.display_label() == "arxiv:2001.00001"

    # W14 was never even upserted into `papers` -- no metadata at all.
    w14 = by_id["W14"]
    assert w14.title is None
    assert w14.doi is None
    assert w14.arxiv_id is None
    assert w14.display_label() == "W14"


def test_compute_gaps_to_dict_shape():
    db = _fixture_db()
    gaps = compute_gaps(db, min_citations=2, top=1)
    d = gaps[0].to_dict()
    assert set(d.keys()) == {"oa_id", "doi", "arxiv_id", "title", "year", "cited_by_count"}
    assert d["oa_id"] == "W10"
    assert d["cited_by_count"] == 3


def test_compute_gaps_empty_db_is_safe():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript(
        """
        CREATE TABLE papers(
          oa_id TEXT PRIMARY KEY, doi TEXT, doc_id TEXT, title TEXT, year INTEGER,
          cited_by_count INTEGER DEFAULT 0, in_corpus INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE cites(
          src_oa TEXT NOT NULL, dst_oa TEXT NOT NULL,
          is_influential INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(src_oa, dst_oa)
        );
        CREATE TABLE paper_aliases(
          provider TEXT NOT NULL, external_id TEXT NOT NULL, oa_id TEXT NOT NULL,
          PRIMARY KEY(provider, external_id)
        );
        """
    )
    assert compute_gaps(db) == []


def test_gap_work_is_frozen_dataclass():
    gw = GapWork(oa_id="W1", doi=None, arxiv_id=None, title=None, year=None, cited_by_count=1)
    assert gw.display_label() == "W1"


# --- --resolve-titles: fetch_openalex_titles / resolve_gap_titles ----------------


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        import json

        return json.dumps(self._payload).encode("utf-8")


def _work(oa_id: str, title: str, year: int, doi: str | None = None) -> dict:
    return {
        "id": f"https://openalex.org/{oa_id}",
        "display_name": title,
        "publication_year": year,
        "ids": {"doi": f"https://doi.org/{doi}"} if doi else {},
    }


def test_fetch_openalex_titles_batches_into_chunks_of_50():
    oa_ids = [f"W{i}" for i in range(120)]  # 3 batches: 50, 50, 20
    calls: list[str] = []

    def _fake_urlopen(req, timeout=30):
        calls.append(req.full_url)
        return _FakeResponse({"results": []})

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        fetch_openalex_titles(oa_ids, mailto="test@example.com", sleep_s=0)

    assert len(calls) == 3
    assert calls[0].count("%7C") + 1 == 50  # 50 ids joined by URL-encoded '|' -> 49 pipes
    assert calls[2].count("%7C") + 1 == 20


def test_fetch_openalex_titles_extracts_title_year_doi_arxiv():
    payload = {
        "results": [
            _work("W10", "Well-Known External Paper", 2015, doi="10.48550/arXiv.2001.00001"),
        ]
    }

    with patch("urllib.request.urlopen", return_value=_FakeResponse(payload)):
        resolved = fetch_openalex_titles(["W10"], mailto="test@example.com", sleep_s=0)

    assert resolved["W10"]["title"] == "Well-Known External Paper"
    assert resolved["W10"]["year"] == 2015
    assert resolved["W10"]["doi"] == "10.48550/arxiv.2001.00001"
    assert resolved["W10"]["arxiv_id"] == "2001.00001"


def test_fetch_openalex_titles_retries_once_on_429_then_succeeds():
    import urllib.error

    payload = {"results": [_work("W10", "Retried Paper", 2019)]}
    attempts = {"n": 0}

    def _fake_urlopen(req, timeout=30):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {}, None)
        return _FakeResponse(payload)

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        with patch("time.sleep") as fake_sleep:
            resolved = fetch_openalex_titles(["W10"], mailto="test@example.com", sleep_s=0)

    assert attempts["n"] == 2
    assert resolved["W10"]["title"] == "Retried Paper"
    fake_sleep.assert_called()  # backoff before the retry


def test_fetch_openalex_titles_degrades_gracefully_on_persistent_failure():
    import urllib.error

    def _fake_urlopen(req, timeout=30):
        raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {}, None)

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        with patch("time.sleep"):
            resolved = fetch_openalex_titles(["W10"], mailto="test@example.com", sleep_s=0)

    assert resolved == {}


def test_fetch_openalex_titles_falls_back_to_singleton_when_batch_endpoint_fails():
    """Live discovery: OpenAlex's `filter=`/search surface is metered separately from
    its plain `/works/{id}` lookup path, and can be rate-limited (budget exhausted)
    while singleton lookups still succeed. The batch call should fall back to one
    singleton lookup per id rather than just giving up."""
    import urllib.error

    def _fake_urlopen(req, timeout=30):
        if "filter=" in req.full_url:
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {}, None)
        # Singleton lookup: /works/W10?...
        return _FakeResponse(_work("W10", "Singleton Fallback Paper", 2017))

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        with patch("time.sleep"):
            resolved = fetch_openalex_titles(["W10"], mailto="test@example.com", sleep_s=0)

    assert resolved["W10"]["title"] == "Singleton Fallback Paper"


def test_fetch_openalex_titles_degrades_gracefully_on_network_error():
    import urllib.error

    def _fake_urlopen(req, timeout=30):
        raise urllib.error.URLError("unreachable")

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        resolved = fetch_openalex_titles(["W10"], mailto="test@example.com", sleep_s=0)

    assert resolved == {}


def test_resolve_gap_titles_writes_back_and_second_call_is_offline():
    db = _fixture_db()
    gaps = compute_gaps(db, min_citations=2, top=50)
    by_id = {g.oa_id: g for g in gaps}
    assert by_id["W14"].title is None  # W14 has no papers row at all

    fetch_calls: list[list[str]] = []

    def _fetch_fn(oa_ids, *, mailto, batch_size, sleep_s):
        fetch_calls.append(list(oa_ids))
        return {
            "W14": {
                "title": "Resolved External Paper",
                "year": 2018,
                "doi": "10.9/w14",
                "arxiv_id": None,
            }
        }

    resolved_gaps = resolve_gap_titles(db, gaps, mailto="test@example.com", fetch_fn=_fetch_fn)
    resolved_by_id = {g.oa_id: g for g in resolved_gaps}
    assert resolved_by_id["W14"].title == "Resolved External Paper"
    assert resolved_by_id["W14"].year == 2018
    assert resolved_by_id["W14"].doi == "10.9/w14"
    assert len(fetch_calls) == 1
    assert "W14" in fetch_calls[0]

    # Persisted: `papers.title` for W14 is now set directly in the store.
    row = db.execute("SELECT title, year, doi FROM papers WHERE oa_id='W14'").fetchone()
    assert row["title"] == "Resolved External Paper"
    assert row["year"] == 2018
    assert row["doi"] == "10.9/w14"

    # Second run: compute_gaps now reads W14's title straight from the store, so
    # resolve_gap_titles must not fetch it again (other still-titleless gaps like
    # W11/W13 would still trigger a fetch -- isolate to just the resolved id).
    gaps_again = compute_gaps(db, min_citations=2, top=50)
    assert next(g.title for g in gaps_again if g.oa_id == "W14") == "Resolved External Paper"

    only_w14 = [g for g in gaps_again if g.oa_id == "W14"]

    def _fetch_fn_should_not_be_called(oa_ids, *, mailto, batch_size, sleep_s):
        raise AssertionError("fetch_fn should not be called when all titles are resolved")

    result = resolve_gap_titles(
        db, only_w14, mailto="test@example.com", fetch_fn=_fetch_fn_should_not_be_called
    )
    assert result == only_w14


def test_resolve_gap_titles_degrades_gracefully_when_fetch_returns_nothing():
    db = _fixture_db()
    gaps = compute_gaps(db, min_citations=2, top=50)

    def _empty_fetch_fn(oa_ids, *, mailto, batch_size, sleep_s):
        return {}

    result = resolve_gap_titles(db, gaps, mailto="test@example.com", fetch_fn=_empty_fetch_fn)
    # Bare ids preserved -- gaps unchanged, no crash.
    assert [g.oa_id for g in result] == [g.oa_id for g in gaps]
    for g, orig in zip(result, gaps, strict=True):
        assert g.title == orig.title
        assert g.arxiv_id == orig.arxiv_id
