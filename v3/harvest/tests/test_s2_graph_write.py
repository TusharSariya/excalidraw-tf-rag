"""Mapper tests: S2Record → S2GraphWrite / s2_writes chunking."""

from __future__ import annotations

from dataclasses import replace

import pytest
from harvest.persistence.models.s2 import (
    SPECTER_V1_DIM,
    S2CitationRow,
    S2PaperRow,
    SourceSemanticScholarRow,
    from_paper_record,
    s2_writes,
)
from harvest.sources.models.s2 import Embedding, RelatedPaper, S2Record


def test_from_paper_record_builds_graph(s2_paper_forceatlas2: dict) -> None:
    # Fixture vector is truncated; strip so mapper does not require 768.
    rec = replace(S2Record.from_dict(s2_paper_forceatlas2), embedding=None)
    write = from_paper_record(rec)

    assert rec.paper_id is not None
    assert write.embedding is None
    assert any(p.paper_id == rec.paper_id for p in write.papers)

    subject = next(p for p in write.papers if p.paper_id == rec.paper_id)
    assert subject.title == rec.title
    assert subject.tldr_text == rec.tldr.text if rec.tldr else None
    assert subject.journal_name == (rec.journal.name if rec.journal else None)
    assert subject.oa_url == (rec.open_access_pdf.url if rec.open_access_pdf else None)
    assert subject.venue_id == (rec.publication_venue.id if rec.publication_venue else None)
    assert subject.fetched_at is not None

    ref_ids = {r.paper_id for r in (rec.references or ()) if r.paper_id}
    cite_ids = {c.paper_id for c in (rec.citations or ()) if c.paper_id}
    paper_ids = {p.paper_id for p in write.papers}
    assert rec.paper_id in paper_ids
    assert ref_ids <= paper_ids
    assert cite_ids <= paper_ids
    assert len(write.papers) == 1 + len(ref_ids | cite_ids)

    assert {e.referenced_paper_id for e in write.references} == ref_ids
    assert all(e.paper_id == rec.paper_id for e in write.references)
    assert {e.citing_paper_id for e in write.citations} == cite_ids
    assert all(e.paper_id == rec.paper_id for e in write.citations)

    assert len(write.authors) >= len(rec.authors or ())
    subject_authorship = [a for a in write.paper_authors if a.paper_id == rec.paper_id]
    assert [a.position for a in subject_authorship] == list(range(len(subject_authorship)))


def test_from_paper_record_requires_paper_id() -> None:
    with pytest.raises(ValueError, match="paper_id is required"):
        from_paper_record(S2Record())


def test_embedding_row_requires_specter_dim(s2_paper_forceatlas2: dict) -> None:
    rec = S2Record.from_dict(s2_paper_forceatlas2)
    assert rec.embedding is not None
    assert rec.embedding.vector is not None
    assert len(rec.embedding.vector) != SPECTER_V1_DIM
    with pytest.raises(ValueError, match=rf"vector\({SPECTER_V1_DIM}\)"):
        from_paper_record(rec)


def test_embedding_row_accepts_specter_dim(s2_paper_forceatlas2: dict) -> None:
    rec = S2Record.from_dict(s2_paper_forceatlas2)
    assert rec.paper_id is not None
    vector = [0.0] * SPECTER_V1_DIM
    rec = replace(
        rec,
        embedding=Embedding(model="specter_v1", vector=vector),
    )
    write = from_paper_record(rec)
    assert write.embedding is not None
    assert write.embedding.paper_id == rec.paper_id
    assert write.embedding.model == "specter_v1"
    assert len(write.embedding.embedding) == SPECTER_V1_DIM


def _stub(paper_id: str) -> RelatedPaper:
    return RelatedPaper(paper_id=paper_id, title=paper_id)


def test_s2_writes_subject_then_edge_chunks_then_outcome() -> None:
    cites = [_stub(f"c{i}") for i in range(2500)]
    refs = [_stub(f"r{i}") for i in range(500)]
    rec = S2Record(
        paper_id="subject",
        title="Subject",
        citation_count=2500,
        reference_count=500,
        citations=cites,
        references=refs,
    )
    writes = s2_writes(42, rec, chunk_size=1000)
    # subject + 3 citation chunks + 1 reference chunk + outcome
    assert len(writes) == 1 + 3 + 1 + 1

    subject_tables = {type(g[0]).__table__ for g in writes[0].groups}
    assert "s2_paper" in subject_tables
    assert "s2_citation" not in subject_tables
    assert "s2_reference" not in subject_tables

    cite_edge_counts = []
    ref_edge_counts = []
    for w in writes[1:-1]:
        tables = {type(g[0]).__table__: len(g) for g in w.groups}
        if "s2_citation" in tables:
            cite_edge_counts.append(tables["s2_citation"])
            assert tables["s2_citation"] <= 1000
            assert all(
                isinstance(r, S2CitationRow)
                for g in w.groups
                if type(g[0]).__table__ == "s2_citation"
                for r in g
            )
        if "s2_reference" in tables:
            ref_edge_counts.append(tables["s2_reference"])
            assert tables["s2_reference"] <= 1000

    assert cite_edge_counts == [1000, 1000, 500]
    assert ref_edge_counts == [500]

    outcome_groups = writes[-1].groups
    assert len(outcome_groups) == 1
    assert isinstance(outcome_groups[0][0], SourceSemanticScholarRow)
    assert outcome_groups[0][0].paper_id == 42
    assert outcome_groups[0][0].status == "succeeded"


def test_s2_writes_empty_related_skips_edge_groups() -> None:
    rec = S2Record(paper_id="subject", title="Solo", citations=[], references=[])
    writes = s2_writes(7, rec)
    assert len(writes) == 2  # subject + outcome only
    assert isinstance(writes[-1].groups[0][0], SourceSemanticScholarRow)
    tables = {type(g[0]).__table__ for w in writes for g in w.groups}
    assert "s2_citation" not in tables
    assert "s2_reference" not in tables
    assert any(isinstance(r, S2PaperRow) for g in writes[0].groups for r in g)
