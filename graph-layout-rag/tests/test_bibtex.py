"""Unit tests for the deterministic BibTeX exporter (no DB, no network)."""

from __future__ import annotations

from graph_layout_rag.citation_store import PaperMeta
from graph_layout_rag.manifest import ManifestItem
from graph_layout_rag.query.bibtex import bibtex_for_doc


def _item(**kw) -> ManifestItem:
    base = dict(
        id="some-doc-id",
        title="Layered Graph Drawing",
        source="arxiv",
        url="https://arxiv.org/abs/1234.5678",
        status="ok",
    )
    base.update(kw)
    return ManifestItem(**base)


def test_full_record_article():
    item = _item(
        authors=["Emden Gansner", "Stephen North"],
        year=1993,
        doi="10.1002/spe.4380211102",
        venue="Software: Practice and Experience",
    )
    bib = bibtex_for_doc(item)
    assert bib.startswith("@article{gansner1993layered,")
    assert "author = {Emden Gansner and Stephen North}," in bib
    assert "year = {1993}," in bib
    assert "doi = {10.1002/spe.4380211102}," in bib
    assert "journal = {Software: Practice and Experience}," in bib
    assert "url = {https://arxiv.org/abs/1234.5678}," in bib
    assert bib.rstrip().endswith("}")


def test_conference_is_inproceedings():
    item = _item(
        authors=["Jane Doe"],
        year=2010,
        venue="Proceedings of the Graph Drawing Symposium",
    )
    bib = bibtex_for_doc(item)
    assert bib.startswith("@inproceedings{")
    assert "booktitle = {Proceedings of the Graph Drawing Symposium}," in bib
    assert "journal" not in bib


def test_missing_authors_omits_author_field():
    item = _item(authors=[], year=2005, venue="Some Journal")
    bib = bibtex_for_doc(item)
    assert "author" not in bib
    # key falls back to year + first title word
    assert bib.startswith("@article{2005layered,")


def test_no_doi_omitted():
    item = _item(authors=["A B"], year=2001, doi=None)
    bib = bibtex_for_doc(item)
    assert "doi" not in bib


def test_unicode_author_key_ascii_value_utf8():
    item = _item(authors=["Émile Zøla"], year=2020, title="Überlayout")
    bib = bibtex_for_doc(item)
    key_line = bib.splitlines()[0]
    # Cite key must be pure ASCII.
    assert key_line.encode("ascii", "ignore").decode("ascii") == key_line
    assert "zla2020uberlayout" in key_line
    # Field value keeps the unicode.
    assert "author = {Émile Zøla}," in bib
    assert "title = {Überlayout}," in bib


def test_minimal_only_title_is_misc_with_safe_key():
    item = _item(authors=[], year=None, title="Stress", url="")
    bib = bibtex_for_doc(item)
    assert bib.startswith("@misc{")
    assert "title = {Stress}," in bib
    # No author/year/doi/url fields.
    assert "author" not in bib and "year" not in bib and "url" not in bib
    # Key derived from first title word (lowercased, ascii).
    key_line = bib.splitlines()[0]
    assert key_line == "@misc{stress,"


def test_authors_fall_back_to_meta_full_authors():
    item = _item(authors=[], year=1999)
    meta = PaperMeta(doc_id="some-doc-id", full_authors=[{"name": "Kozo Sugiyama"}])
    bib = bibtex_for_doc(item, meta)
    assert "author = {Kozo Sugiyama}," in bib
    assert bib.startswith("@misc{sugiyama1999layered,") or "sugiyama1999" in bib
