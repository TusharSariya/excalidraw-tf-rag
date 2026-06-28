import math

from graph_layout_rag import citation_store as cs


def test_normalize_doi():
    assert cs.normalize_doi("https://doi.org/10.1109/TSE.1979.234212") == "10.1109/tse.1979.234212"
    assert cs.normalize_doi("doi:10.1007/BF02122694") == "10.1007/bf02122694"
    assert cs.normalize_doi("10.1145/321850.321852).") == "10.1145/321850.321852"
    assert cs.normalize_doi("not-a-doi") is None
    assert cs.normalize_doi(None) is None


def test_normalize_oa_id():
    assert cs.normalize_oa_id("https://openalex.org/W2034567") == "W2034567"
    assert cs.normalize_oa_id("W123") == "W123"
    assert cs.normalize_oa_id(None) is None
    assert cs.normalize_oa_id("https://openalex.org/A555") is None  # author id, not a work


def _db(tmp_path, monkeypatch):
    path = tmp_path / "citations.sqlite"
    monkeypatch.setattr("graph_layout_rag.citation_store.CITATIONS_DB_PATH", path)
    return cs.connect(path)


def test_upsert_paper_idempotent_and_preserves(tmp_path, monkeypatch):
    db = _db(tmp_path, monkeypatch)
    cs.upsert_paper(db, oa_id="W1", doi="10.1/a", doc_id="doc-a", title="A", year=1990,
                    cited_by_count=100, in_corpus=True, enriched_at="t0")
    # A cheap external-neighbor re-upsert (nulls) must not clobber the full row.
    cs.upsert_paper(db, oa_id="W1")
    row = cs.paper_row(db, "W1")
    assert row["doi"] == "10.1/a" and row["doc_id"] == "doc-a"
    assert row["cited_by_count"] == 100 and row["in_corpus"] == 1
    # cited_by_count only grows.
    cs.upsert_paper(db, oa_id="W1", cited_by_count=50)
    assert cs.paper_row(db, "W1")["cited_by_count"] == 100


def test_cites_and_influential_merge(tmp_path, monkeypatch):
    db = _db(tmp_path, monkeypatch)
    cs.upsert_paper(db, oa_id="W1", doi="10.1/a", doc_id="doc-a", in_corpus=True, enriched_at="t")
    cs.upsert_paper(db, oa_id="W2", doi="10.1/b", doc_id="doc-b", in_corpus=True, enriched_at="t")
    cs.add_cites(db, [("W1", "W2", 0), ("W1", "W1", 0)])  # self-loop dropped
    assert cs.references_of(db, "W1") == {"W2"}
    assert cs.cited_by_of(db, "W2") == {"W1"}
    # influential flag is OR-ed in and never reset.
    assert cs.set_influential_by_doi(db, "10.1/a", "10.1/b") == 1
    assert db.execute("SELECT is_influential FROM cites WHERE src_oa='W1'").fetchone()[0] == 1
    cs.add_cites(db, [("W1", "W2", 0)])
    assert db.execute("SELECT is_influential FROM cites WHERE src_oa='W1'").fetchone()[0] == 1


def test_authorship_and_counts(tmp_path, monkeypatch):
    db = _db(tmp_path, monkeypatch)
    cs.upsert_paper(db, oa_id="W1", doi="10.1/a", doc_id="doc-a", in_corpus=True, enriched_at="t")
    cs.add_authorships(db, [(cs.author_key("Eades, P."), "doc-a"),
                            (cs.author_key("Peter Eades"), "doc-b")])
    # "Eades, P." and "Peter Eades" don't collapse, but exact repeats are ignored.
    cs.add_authorships(db, [(cs.author_key("Sugiyama"), "doc-a"),
                            (cs.author_key("Sugiyama"), "doc-b")])
    assert cs.coauthored_doc_ids(db, "doc-a") == {"doc-b"}  # shared "sugiyama"
    c = cs.counts(db)
    assert c["corpus_papers"] == 1 and c["cite_edges"] == 0 and c["authorship_edges"] == 4


def test_aliases_and_citation_provenance(tmp_path, monkeypatch):
    db = _db(tmp_path, monkeypatch)
    cs.upsert_paper(db, oa_id="W1", doi="10.1/a")
    cs.upsert_paper(db, oa_id="W2", doi="10.1/b")
    cs.upsert_alias(db, provider="DOI", external_id="10.1/a", oa_id="W1")
    cs.add_cites(db, [("W1", "W2", 0)], provider="openalex")
    cs.add_cites(db, [("W1", "W2", 1)], provider="semantic-scholar")
    db.commit()

    assert cs.oa_id_for_alias(db, provider="doi", external_id="10.1/a") == "W1"
    providers = {
        row[0]
        for row in db.execute(
            "SELECT provider FROM cite_provenance WHERE src_oa='W1' AND dst_oa='W2'"
        )
    }
    assert providers == {"openalex", "semantic-scholar"}
    assert cs.counts(db)["citation_provenance"] == 2


def test_paper_meta_roundtrip_full_and_partial(tmp_path, monkeypatch):
    db = _db(tmp_path, monkeypatch)
    cs.upsert_paper_meta(
        db,
        "doc-a",
        tldr="short summary",
        abstract="long abstract",
        fwci=1.5,
        cited_by_count=42,
        oa_pdf_url="https://example.test/a.pdf",
        license="cc-by",
        biblio={"volume": "3", "issue": "1"},
        full_authors=[{"name": "Eades"}, {"name": "Sugiyama"}],
        source_provider="openalex",
        enriched_at="t0",
    )
    meta = cs.paper_meta_for_doc(db, "doc-a")
    assert meta is not None
    assert meta.doc_id == "doc-a"
    assert meta.tldr == "short summary"
    assert meta.abstract == "long abstract"
    assert meta.fwci == 1.5
    assert meta.cited_by_count == 42
    assert meta.oa_pdf_url == "https://example.test/a.pdf"
    assert meta.license == "cc-by"
    assert meta.biblio == {"volume": "3", "issue": "1"}
    assert meta.full_authors == [{"name": "Eades"}, {"name": "Sugiyama"}]
    assert meta.source_provider == "openalex"
    assert meta.enriched_at == "t0"

    # Partial record: only a doc_id; JSON columns stay None.
    cs.upsert_paper_meta(db, "doc-b", tldr="b")
    mb = cs.paper_meta_for_doc(db, "doc-b")
    assert mb is not None and mb.tldr == "b"
    assert mb.biblio is None and mb.full_authors is None
    assert mb.enriched_at  # auto-stamped when not supplied


def test_paper_meta_missing_returns_none(tmp_path, monkeypatch):
    db = _db(tmp_path, monkeypatch)
    assert cs.paper_meta_for_doc(db, "nope") is None
    assert cs.has_paper_meta(db, "nope") is False


def test_paper_meta_map_bulk(tmp_path, monkeypatch):
    db = _db(tmp_path, monkeypatch)
    cs.upsert_paper_meta(db, "doc-a", tldr="a")
    cs.upsert_paper_meta(db, "doc-b", tldr="b")
    all_map = cs.paper_meta_map(db)
    assert set(all_map) == {"doc-a", "doc-b"}
    subset = cs.paper_meta_map(db, ["doc-a", "missing"])
    assert set(subset) == {"doc-a"}
    assert cs.has_paper_meta(db, "doc-a") is True


def test_paper_meta_coalesce_partial_update_does_not_null(tmp_path, monkeypatch):
    db = _db(tmp_path, monkeypatch)
    cs.upsert_paper_meta(
        db, "doc-a", tldr="keep", abstract="keep-abs", fwci=2.0,
        biblio={"volume": "1"}, full_authors=[{"name": "X"}],
    )
    # Later partial write touches only cited_by_count; existing columns survive.
    cs.upsert_paper_meta(db, "doc-a", cited_by_count=99)
    meta = cs.paper_meta_for_doc(db, "doc-a")
    assert meta.tldr == "keep"
    assert meta.abstract == "keep-abs"
    assert meta.fwci == 2.0
    assert meta.biblio == {"volume": "1"}
    assert meta.full_authors == [{"name": "X"}]
    assert meta.cited_by_count == 99


def test_load_paper_meta_cached_empty_when_db_absent(tmp_path, monkeypatch):
    missing = tmp_path / "does-not-exist.sqlite"
    monkeypatch.setattr("graph_layout_rag.citation_store.CITATIONS_DB_PATH", missing)
    cs.load_paper_meta_cached.cache_clear()
    try:
        assert cs.load_paper_meta_cached() == {}
    finally:
        cs.load_paper_meta_cached.cache_clear()


def test_specter2_roundtrip(tmp_path, monkeypatch):
    db = _db(tmp_path, monkeypatch)
    vec = [0.1, -0.25, 3.5, 0.0]
    cs.upsert_specter2(db, "doc-a", vec)
    out = cs.specter2_for_doc(db, "doc-a")
    assert out is not None and len(out) == len(vec)
    for a, b in zip(out, vec):
        assert math.isclose(a, b, rel_tol=0, abs_tol=1e-6)
    assert db.execute("SELECT dim FROM doc_specter2 WHERE doc_id='doc-a'").fetchone()[0] == 4
    assert cs.specter2_for_doc(db, "missing") is None
