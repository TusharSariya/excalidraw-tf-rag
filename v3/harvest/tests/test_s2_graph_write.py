"""Mapper tests: PaperRecord → S2GraphWrite."""

from __future__ import annotations

from dataclasses import replace

import pytest
from harvest.persistence.models.s2 import SPECTER_V1_DIM, from_paper_record
from harvest.sources.models.s2 import Embedding, PaperRecord


def test_from_paper_record_builds_graph(s2_paper_forceatlas2: dict) -> None:
    # Fixture vector is truncated; strip so mapper does not require 768.
    rec = replace(PaperRecord.from_dict(s2_paper_forceatlas2), embedding=None)
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
        from_paper_record(PaperRecord())


def test_embedding_row_requires_specter_dim(s2_paper_forceatlas2: dict) -> None:
    rec = PaperRecord.from_dict(s2_paper_forceatlas2)
    assert rec.embedding is not None
    assert rec.embedding.vector is not None
    assert len(rec.embedding.vector) != SPECTER_V1_DIM
    with pytest.raises(ValueError, match=rf"vector\({SPECTER_V1_DIM}\)"):
        from_paper_record(rec)


def test_embedding_row_accepts_specter_dim(s2_paper_forceatlas2: dict) -> None:
    rec = PaperRecord.from_dict(s2_paper_forceatlas2)
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
