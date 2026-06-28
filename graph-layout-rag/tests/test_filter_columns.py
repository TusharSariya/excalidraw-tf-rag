"""T5/P2: the six metadata fields (venue, arxiv_category, genre, venue_type,
oa_version, is_retracted) are threaded from the manifest item into chunk rows as
FILTER-ONLY index columns, and the retrieve path PRE-filters on them.

Critical invariant: these fields must NOT enter the embedded/BM25 text (embedding
metadata is A/B-proven NULL on this corpus and would cache-bust every chunk).
"""
from __future__ import annotations

from pathlib import Path

from graph_layout_rag.ingest import bm25
from graph_layout_rag.ingest.chunk import (
    chunk_metadata,
    embed_body_text,
    embed_input_text,
)
from graph_layout_rag.ingest.index import _chunk_row
from graph_layout_rag.manifest import ManifestItem
from graph_layout_rag.query import retrieve as retrieve_mod
from graph_layout_rag.query.retrieve import (
    RetrieveFilters,
    _apply_filters,
    _dense_where,
    _pool_size,
)

_SENTINEL_VENUE = "Journal of Sentinel Venues O'Hare"


def _item(**kw) -> ManifestItem:
    base = dict(
        id="doc1",
        title="Layered Graph Drawing",
        authors=["Sugiyama"],
        year=1981,
        source="handbook",
        url="https://example.org/doc1",
        status="ok",
        tags=["sugiyama"],
        venue=_SENTINEL_VENUE,
        arxiv_category="cs.DS",
        genre="JournalArticle",
        venue_type="journal",
        oa_version="publishedVersion",
        is_retracted=True,
    )
    base.update(kw)
    return ManifestItem(**base)


# --- chunk.py: fields populate from the manifest item, absent from embed text ---


def test_make_chunk_populates_filter_fields():
    chunk = chunk_metadata(_item(), "body text")[0]
    assert chunk.venue == _SENTINEL_VENUE
    assert chunk.arxiv_category == "cs.DS"
    assert chunk.genre == "JournalArticle"
    assert chunk.venue_type == "journal"
    assert chunk.oa_version == "publishedVersion"
    assert chunk.is_retracted is True


def test_filter_fields_not_in_embed_text():
    chunk = chunk_metadata(_item(), "body text")[0]
    for text in (embed_input_text(chunk), embed_body_text(chunk)):
        assert _SENTINEL_VENUE not in text
        assert "cs.DS" not in text
        assert "JournalArticle" not in text
        assert "publishedVersion" not in text


def test_filter_fields_default_when_absent():
    item = ManifestItem(
        id="d", title="t", source="s", url="u", status="ok",
    )
    chunk = chunk_metadata(item, "body")[0]
    assert chunk.venue is None
    assert chunk.is_retracted is False


# --- index.py: _chunk_row carries the six keys with the right types -------------


def test_chunk_row_has_filter_columns_with_types():
    chunk = chunk_metadata(_item(), "body text")[0]
    row = _chunk_row(chunk, [0.0, 0.1])
    assert row["venue"] == _SENTINEL_VENUE
    assert row["arxiv_category"] == "cs.DS"
    assert row["genre"] == "JournalArticle"
    assert row["venue_type"] == "journal"
    assert row["oa_version"] == "publishedVersion"
    # is_retracted is stored as an int 0/1 (no boolean column in the row schema).
    assert row["is_retracted"] == 1
    assert isinstance(row["is_retracted"], int)
    for key in ("venue", "arxiv_category", "genre", "venue_type", "oa_version"):
        assert isinstance(row[key], str)


def test_chunk_row_empty_strings_when_absent():
    item = ManifestItem(id="d", title="t", source="s", url="u", status="ok")
    row = _chunk_row(chunk_metadata(item, "body")[0], [0.0])
    assert row["venue"] == ""
    assert row["is_retracted"] == 0


# --- retrieve.py: _dense_where builds AND-ed, quote-escaped predicates ----------


def test_dense_where_none_when_no_filters():
    assert _dense_where(RetrieveFilters()) is None


def test_dense_where_year_only_matches_legacy_shape():
    where = _dense_where(RetrieveFilters(year_min=2000))
    assert where == "(year >= 2000 OR year IS NULL)"


def test_dense_where_single_string_filter():
    where = _dense_where(RetrieveFilters(genre="JournalArticle"))
    assert where == "genre = 'JournalArticle'"


def test_dense_where_exclude_retracted():
    where = _dense_where(RetrieveFilters(exclude_retracted=True))
    assert where == "(is_retracted = 0 OR is_retracted IS NULL)"


def test_dense_where_escapes_apostrophes():
    where = _dense_where(RetrieveFilters(venue="O'Hare Press"))
    assert where == "venue = 'O''Hare Press'"


def test_dense_where_ands_all_clauses():
    where = _dense_where(
        RetrieveFilters(
            year_min=2010,
            venue="ACM",
            arxiv_category="cs.DS",
            genre="JournalArticle",
            exclude_retracted=True,
        )
    )
    assert where == (
        "(year >= 2010 OR year IS NULL) AND "
        "venue = 'ACM' AND "
        "arxiv_category = 'cs.DS' AND "
        "genre = 'JournalArticle' AND "
        "(is_retracted = 0 OR is_retracted IS NULL)"
    )


# --- _pool_size: the new filters mark the pool selective ------------------------


def test_pool_size_unfiltered_is_baseline():
    assert _pool_size(top=8, filters=RetrieveFilters()) == max(80, 8 * 4)


def test_pool_size_new_filters_widen_pool():
    for f in (
        RetrieveFilters(venue="ACM"),
        RetrieveFilters(arxiv_category="cs.DS"),
        RetrieveFilters(genre="JournalArticle"),
        RetrieveFilters(exclude_retracted=True),
    ):
        assert _pool_size(top=8, filters=f) == max(80, 8 * 12)


# --- _apply_filters: post-filter / sparse-side honors the new fields ------------


def _row(**kw):
    base = {"doc_id": "d", "venue": "", "arxiv_category": "", "genre": "", "is_retracted": 0}
    base.update(kw)
    return base


def test_apply_filters_venue_match_and_miss():
    rows = [_row(doc_id="a", venue="ACM"), _row(doc_id="b", venue="IEEE")]
    out = _apply_filters(rows, RetrieveFilters(venue="ACM"))
    assert [r["doc_id"] for r in out] == ["a"]


def test_apply_filters_exclude_retracted():
    rows = [_row(doc_id="a", is_retracted=1), _row(doc_id="b", is_retracted=0)]
    out = _apply_filters(rows, RetrieveFilters(exclude_retracted=True))
    assert [r["doc_id"] for r in out] == ["b"]


# --- BM25: the new fields are indexed and filterable (round-trip) ---------------


def test_bm25_roundtrips_filter_fields(tmp_path: Path):
    chunk = chunk_metadata(
        _item(id="a"), "network simplex rank assignment in layered drawing"
    )[0]
    bm25.upsert_chunks([chunk], [chunk.text], index_dir=tmp_path / "bm25", rebuild=True)
    hits = bm25.search_bm25("network simplex", index_dir=tmp_path / "bm25", limit=10)
    assert hits
    hit = hits[0]
    assert hit["venue"] == _SENTINEL_VENUE
    assert hit["arxiv_category"] == "cs.DS"
    assert hit["genre"] == "JournalArticle"
    assert hit["venue_type"] == "journal"
    assert hit["oa_version"] == "publishedVersion"
    assert hit["is_retracted"] == 1
    # The sparse text must NOT carry the filter metadata.
    assert _SENTINEL_VENUE not in hit["text"]


def test_bm25_results_are_filterable_via_apply_filters(tmp_path: Path):
    a = chunk_metadata(_item(id="a", venue="ACM"), "layered drawing one")[0]
    b = chunk_metadata(_item(id="b", venue="IEEE"), "layered drawing two")[0]
    bm25.upsert_chunks([a, b], [a.text, b.text], index_dir=tmp_path / "bm25", rebuild=True)
    hits = bm25.search_bm25("layered drawing", index_dir=tmp_path / "bm25", limit=10)
    filtered = _apply_filters(hits, RetrieveFilters(venue="ACM"))
    assert filtered
    assert all(r["venue"] == "ACM" for r in filtered)


# --- Core Codex fix: selective venue PRE-filters in the dense WHERE, so a match
#     OUTSIDE the naive top pool is still found (not post-pool filtered). ---------


class _FakeDenseSearch:
    """Records the WHERE predicate and returns only rows matching it (simulating a
    LanceDB prefilter that scans the whole table, not just a naive top-N pool)."""

    def __init__(self, rows):
        self.rows = rows
        self.where_clause = None
        self.prefilter = None

    def metric(self, _m):
        return self

    def where(self, clause, prefilter=False):
        self.where_clause = clause
        self.prefilter = prefilter
        return self

    def limit(self, _n):
        return self

    def to_list(self):
        # Emulate LanceDB prefilter: a venue='ACM' WHERE returns the ACM row even
        # though it would rank far below the pool cutoff in an unfiltered search.
        if self.where_clause and "venue = 'ACM'" in self.where_clause:
            return [dict(r) for r in self.rows if r.get("venue") == "ACM"]
        return [dict(r) for r in self.rows]


class _FakeTable:
    def __init__(self, rows):
        self._search = _FakeDenseSearch(rows)

    def search(self, _vec):
        return self._search


def test_selective_venue_prefilters_in_dense_where(monkeypatch):
    # A single ACM doc buried under many IEEE docs; an unfiltered top-N pool would
    # drop it, but the venue='ACM' prefilter surfaces it.
    rows = [_row(doc_id=f"ieee{i}", venue="IEEE", _distance=0.1) for i in range(50)]
    rows.append(_row(doc_id="acm1", venue="ACM", _distance=0.9))
    table = _FakeTable(rows)

    out = retrieve_mod._dense_search(
        table, [0.0, 0.1], pool=80, filters=RetrieveFilters(venue="ACM")
    )
    assert table._search.prefilter is True
    assert "venue = 'ACM'" in table._search.where_clause
    assert [r["doc_id"] for r in out] == ["acm1"]
