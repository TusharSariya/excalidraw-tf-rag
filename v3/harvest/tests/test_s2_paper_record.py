"""Fixture-driven mapping tests for models.s2.PaperRecord."""

from __future__ import annotations

import pytest
from harvest.sources.models.s2 import TOP_LEVEL_KEYS, PaperRecord, RelatedPaper


def test_fixture_has_all_top_level_keys(s2_paper_forceatlas2: dict) -> None:
    assert set(s2_paper_forceatlas2.keys()) == TOP_LEVEL_KEYS


def test_paper_record_maps_all_top_level_keys(s2_paper_forceatlas2: dict) -> None:
    rec = PaperRecord.from_dict(s2_paper_forceatlas2)

    assert rec.paper_id == s2_paper_forceatlas2["paperId"]
    assert rec.title == s2_paper_forceatlas2["title"]
    assert rec.corpus_id == s2_paper_forceatlas2["corpusId"]
    assert rec.abstract == s2_paper_forceatlas2["abstract"]
    assert rec.venue == s2_paper_forceatlas2["venue"]
    assert rec.year == s2_paper_forceatlas2["year"]
    assert rec.url == s2_paper_forceatlas2["url"]
    assert rec.reference_count == s2_paper_forceatlas2["referenceCount"]
    assert rec.citation_count == s2_paper_forceatlas2["citationCount"]
    assert rec.influential_citation_count == s2_paper_forceatlas2["influentialCitationCount"]
    assert rec.is_open_access == s2_paper_forceatlas2["isOpenAccess"]
    assert rec.publication_date == s2_paper_forceatlas2["publicationDate"]
    assert rec.publication_types == s2_paper_forceatlas2["publicationTypes"]
    assert rec.fields_of_study == s2_paper_forceatlas2["fieldsOfStudy"]
    assert rec.external_ids == s2_paper_forceatlas2["externalIds"]
    assert rec.citation_styles == s2_paper_forceatlas2["citationStyles"]

    assert rec.authors is not None and len(rec.authors) == len(s2_paper_forceatlas2["authors"])
    assert rec.authors[0].name == s2_paper_forceatlas2["authors"][0]["name"]
    assert rec.authors[0].author_id == s2_paper_forceatlas2["authors"][0]["authorId"]
    assert rec.authors[0].h_index == s2_paper_forceatlas2["authors"][0]["hIndex"]

    assert rec.journal is not None
    assert rec.journal.name == s2_paper_forceatlas2["journal"]["name"]
    assert rec.journal.volume == s2_paper_forceatlas2["journal"]["volume"]

    assert rec.open_access_pdf is not None
    assert rec.open_access_pdf.url == s2_paper_forceatlas2["openAccessPdf"]["url"]
    assert rec.open_access_pdf.status == s2_paper_forceatlas2["openAccessPdf"]["status"]

    assert rec.publication_venue is not None
    assert rec.publication_venue.id == s2_paper_forceatlas2["publicationVenue"]["id"]
    assert rec.publication_venue.issn == s2_paper_forceatlas2["publicationVenue"]["issn"]

    assert rec.s2_fields_of_study is not None
    assert (
        rec.s2_fields_of_study[0].category == s2_paper_forceatlas2["s2FieldsOfStudy"][0]["category"]
    )
    assert rec.s2_fields_of_study[0].source == s2_paper_forceatlas2["s2FieldsOfStudy"][0]["source"]

    assert rec.tldr is not None
    assert rec.tldr.text == s2_paper_forceatlas2["tldr"]["text"]
    assert rec.tldr.model == s2_paper_forceatlas2["tldr"]["model"]

    assert rec.embedding is not None
    assert rec.embedding.model == s2_paper_forceatlas2["embedding"]["model"]
    assert rec.embedding.vector is not None
    assert len(rec.embedding.vector) == len(s2_paper_forceatlas2["embedding"]["vector"])

    assert rec.citations is not None
    assert len(rec.citations) == len(s2_paper_forceatlas2["citations"])
    assert rec.citations[0].title == s2_paper_forceatlas2["citations"][0]["title"]
    assert rec.citations[0].paper_id == s2_paper_forceatlas2["citations"][0]["paperId"]

    assert rec.references is not None
    assert len(rec.references) == len(s2_paper_forceatlas2["references"])
    assert rec.references[0].title == s2_paper_forceatlas2["references"][0]["title"]


def test_from_dict_rejects_unknown_top_level_key(s2_paper_forceatlas2: dict) -> None:
    bad = {**s2_paper_forceatlas2, "notARealField": 1}
    with pytest.raises(ValueError, match="unknown top-level field"):
        PaperRecord.from_dict(bad)


def test_related_paper_rejects_nested_citations() -> None:
    with pytest.raises(ValueError, match="unknown field"):
        RelatedPaper.from_dict({"paperId": "x", "citations": []})
