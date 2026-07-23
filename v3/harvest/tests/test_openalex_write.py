"""openalex mapping: OpenAlexWork -> source_openalex row, and key normalisation."""

from harvest.persistence.models import openalex_write
from harvest.sources.models.openalex import OpenAlexWork
from harvest.sources.openalex import _normalize_key

FIXTURE = {
    "id": "https://openalex.org/W123",
    "title": "A Test Work",
    "doi": "https://doi.org/10.1/x",
    "publication_date": "2020-05-01",
    "publication_year": 2020,
    "language": "en",
    "type": "article",
    "best_oa_location": {"pdf_url": "https://example.org/paper.pdf"},
    "open_access": {"is_oa": True, "oa_status": "green"},
    "cited_by_count": 7,
    "referenced_works_count": 3,
    "is_retracted": False,
    "authorships": [{"author": {"display_name": "Jane Doe"}, "author_position": "first"}],
    "referenced_works": ["https://openalex.org/W9"],
}


def test_openalex_write_maps_a_source_openalex_row():
    write = openalex_write(42, OpenAlexWork.from_result(FIXTURE))
    assert len(write.groups) == 1 and len(write.groups[0]) == 1
    row = write.groups[0][0]
    assert row.__table__ == "source_openalex"
    assert row.__conflict__ == ("paper_id",)

    d = row.model_dump(mode="json")
    assert d["paper_id"] == 42
    assert d["status"] == "succeeded"
    assert d["openalex_id"] == "https://openalex.org/W123"
    assert d["doi"] == "https://doi.org/10.1/x"
    assert d["publication_year"] == 2020
    assert d["language"] == "en"
    assert d["type"] == "article"
    assert d["best_oa_pdf_url"] == "https://example.org/paper.pdf"
    assert d["is_oa"] is True
    assert d["oa_status"] == "green"
    assert d["cited_by_count"] == 7
    assert d["referenced_works_count"] == 3
    assert d["is_retracted"] is False
    assert d["authorships"] == FIXTURE["authorships"]
    assert d["fetched_at"] is not None
    # raw keeps the whole record, incl. the citation ids not broken out to a column
    assert d["raw"]["referenced_works"] == ["https://openalex.org/W9"]


def test_openalex_write_tolerates_a_sparse_work():
    """OpenAlex omits fields freely — missing open_access / pdf must not blow up."""
    row = openalex_write(1, OpenAlexWork.from_result({"id": "W1", "title": "T"})).groups[0][0]
    d = row.model_dump(mode="json")
    assert d["is_oa"] is None
    assert d["oa_status"] is None
    assert d["best_oa_pdf_url"] is None
    assert d["title"] == "T"


def test_normalize_key_normalises_identifiers():
    assert (
        _normalize_key("10.1109/TVCG.2021.3114756") == "https://doi.org/10.1109/TVCG.2021.3114756"
    )
    assert _normalize_key("W2741809807") == "W2741809807"
    assert _normalize_key("https://openalex.org/W1") == "https://openalex.org/W1"
    assert _normalize_key("pmid:12345") == "pmid:12345"  # namespaced ids pass through


def test_normalize_key_normalises_ror_even_though_inert():
    # ROR is institution-scoped and 404s for works, but is normalised to the namespace.
    assert _normalize_key("https://ror.org/0abc12345") == "ror:0abc12345"
    assert _normalize_key("ror:0abc12345") == "ror:0abc12345"
