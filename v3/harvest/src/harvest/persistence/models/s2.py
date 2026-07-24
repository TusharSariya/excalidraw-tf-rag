"""Table-shaped DTOs for public.s2_* + S2Record → S2GraphWrite mapper."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from harvest.persistence.base import TableRow, Write
from harvest.sources.models.s2 import (
    Author,
    Embedding,
    RelatedPaper,
    S2Record,
)
from pydantic import Field

#: specter_v1 width; must match extensions.vector(768) on s2_embedding.
SPECTER_V1_DIM = 768

#: Max related papers (citations or references) per PostgREST upsert Write.
#: Keeps statement size / worker RSS bounded for highly-cited subjects.
S2_EDGE_UPSERT_CHUNK = 1000


class S2PaperRow(TableRow):
    """Insert/upsert shape for public.s2_paper."""

    __table__ = "s2_paper"
    __conflict__ = ("paper_id",)

    paper_id: str
    corpus_id: int | None = None
    title: str | None = None
    abstract: str | None = None
    venue: str | None = None
    year: int | None = None
    url: str | None = None
    reference_count: int | None = None
    citation_count: int | None = None
    influential_citation_count: int | None = None
    is_open_access: bool | None = None
    publication_date: str | None = None
    publication_types: list[str] | None = None
    fields_of_study: list[str] | None = None
    external_ids: dict[str, Any] | None = None
    citation_styles: dict[str, str] | None = None
    journal_name: str | None = None
    journal_volume: str | None = None
    journal_pages: str | None = None
    oa_url: str | None = None
    oa_status: str | None = None
    oa_license: str | None = None
    oa_disclaimer: str | None = None
    venue_id: str | None = None
    venue_name: str | None = None
    venue_type: str | None = None
    venue_issn: str | None = None
    venue_url: str | None = None
    venue_alternate_names: list[str] | None = None
    venue_alternate_urls: list[str] | None = None
    tldr_model: str | None = None
    tldr_text: str | None = None
    s2_fields_of_study: list[dict[str, str | None]] | None = None
    fetched_at: datetime | None = None


class S2AuthorRow(TableRow):
    """Insert/upsert shape for public.s2_author."""

    __table__ = "s2_author"
    __conflict__ = ("author_id",)

    author_id: str
    name: str | None = None
    url: str | None = None
    affiliations: list[str] | None = None
    homepage: str | None = None
    paper_count: int | None = None
    citation_count: int | None = None
    h_index: int | None = None
    external_ids: dict[str, Any] | None = None


class S2PaperAuthorRow(TableRow):
    """Insert/upsert shape for public.s2_paper_author."""

    __table__ = "s2_paper_author"
    __conflict__ = ("paper_id", "author_id")

    paper_id: str
    author_id: str
    position: int = Field(ge=0)


class S2ReferenceRow(TableRow):
    """Insert/upsert shape for public.s2_reference."""

    __table__ = "s2_reference"
    __conflict__ = ("paper_id", "referenced_paper_id")

    paper_id: str
    referenced_paper_id: str


class S2CitationRow(TableRow):
    """Insert/upsert shape for public.s2_citation."""

    __table__ = "s2_citation"
    __conflict__ = ("paper_id", "citing_paper_id")

    paper_id: str
    citing_paper_id: str


class S2EmbeddingRow(TableRow):
    """Insert/upsert shape for public.s2_embedding (vector(768))."""

    __table__ = "s2_embedding"
    __conflict__ = ("paper_id",)

    paper_id: str
    model: str
    embedding: list[float]
    fetched_at: datetime | None = None


class SourceSemanticScholarRow(TableRow):
    """Insert/upsert shape for public.source_semantic_scholar (per-seed outcome).

    Keyed by the seed's bigint ``paper_id`` — a different namespace from the
    ``s2_*`` graph tables, whose ``paper_id`` is S2's own text id (kept here as
    ``s2_paper_id``). One row per seed records whether the S2 fetch succeeded.
    """

    __table__ = "source_semantic_scholar"
    __conflict__ = ("paper_id",)

    paper_id: int
    status: str = "succeeded"
    s2_paper_id: str | None = None
    corpus_id: int | None = None
    title: str | None = None
    external_ids: dict[str, Any] | None = None
    open_access_pdf_url: str | None = None
    reference_count: int | None = None
    citation_count: int | None = None
    fields_of_study: list[str] | None = None
    raw: dict[str, Any] | None = None
    error: str | None = None
    fetched_at: datetime | None = None


@dataclass(frozen=True)
class S2GraphWrite:
    """Multi-table upsert payload derived from one S2Record."""

    papers: list[S2PaperRow]
    authors: list[S2AuthorRow]
    paper_authors: list[S2PaperAuthorRow]
    references: list[S2ReferenceRow]
    citations: list[S2CitationRow]
    embedding: S2EmbeddingRow | None

    def to_write(self) -> Write:
        """Flatten into FK-ordered row-groups for a generic backend.

        s2_paper (subject + every reference/citation stub) and s2_author come
        first, so the paper_author / reference / citation / embedding rows that
        FK into them never precede their parents. Empty groups are dropped so no
        empty upsert is issued.
        """
        groups: list[Sequence[TableRow]] = [
            self.papers,
            self.authors,
            self.paper_authors,
            self.references,
            self.citations,
        ]
        if self.embedding is not None:
            groups.append([self.embedding])
        return Write(groups=[g for g in groups if g])


def _fos_json(
    fields: list[Any] | None,
) -> list[dict[str, str | None]] | None:
    if fields is None:
        return None
    return [{"category": f.category, "source": f.source} for f in fields]


def paper_row_from(
    paper: S2Record | RelatedPaper,
    *,
    fetched_at: datetime | None = None,
    include_tldr: bool = False,
) -> S2PaperRow | None:
    """Flatten a typed paper into an s2_paper row. None if paper_id missing."""
    if not paper.paper_id:
        return None

    journal = paper.journal
    oa = paper.open_access_pdf
    venue = paper.publication_venue
    tldr_model = tldr_text = None
    if include_tldr and isinstance(paper, S2Record) and paper.tldr is not None:
        tldr_model = paper.tldr.model
        tldr_text = paper.tldr.text

    return S2PaperRow(
        paper_id=paper.paper_id,
        corpus_id=paper.corpus_id,
        title=paper.title,
        abstract=paper.abstract,
        venue=paper.venue,
        year=paper.year,
        url=paper.url,
        reference_count=paper.reference_count,
        citation_count=paper.citation_count,
        influential_citation_count=paper.influential_citation_count,
        is_open_access=paper.is_open_access,
        publication_date=paper.publication_date,
        publication_types=paper.publication_types,
        fields_of_study=paper.fields_of_study,
        external_ids=paper.external_ids,
        citation_styles=paper.citation_styles,
        journal_name=journal.name if journal else None,
        journal_volume=journal.volume if journal else None,
        journal_pages=journal.pages if journal else None,
        oa_url=oa.url if oa else None,
        oa_status=oa.status if oa else None,
        oa_license=oa.license if oa else None,
        oa_disclaimer=oa.disclaimer if oa else None,
        venue_id=venue.id if venue else None,
        venue_name=venue.name if venue else None,
        venue_type=venue.type if venue else None,
        venue_issn=venue.issn if venue else None,
        venue_url=venue.url if venue else None,
        venue_alternate_names=venue.alternate_names if venue else None,
        venue_alternate_urls=venue.alternate_urls if venue else None,
        tldr_model=tldr_model,
        tldr_text=tldr_text,
        s2_fields_of_study=_fos_json(paper.s2_fields_of_study),
        fetched_at=fetched_at,
    )


def author_row_from(author: Author) -> S2AuthorRow | None:
    if not author.author_id:
        return None
    return S2AuthorRow(
        author_id=author.author_id,
        name=author.name,
        url=author.url,
        affiliations=author.affiliations,
        homepage=author.homepage,
        paper_count=author.paper_count,
        citation_count=author.citation_count,
        h_index=author.h_index,
        external_ids=author.external_ids,
    )


def embedding_row_from(
    paper_id: str,
    embedding: Embedding | None,
    *,
    fetched_at: datetime | None = None,
) -> S2EmbeddingRow | None:
    if embedding is None or embedding.model is None or embedding.vector is None:
        return None
    dim = len(embedding.vector)
    if dim != SPECTER_V1_DIM:
        raise ValueError(
            f"s2_embedding expects vector({SPECTER_V1_DIM}), got dim={dim} "
            f"(model={embedding.model!r})"
        )
    return S2EmbeddingRow(
        paper_id=paper_id,
        model=embedding.model,
        embedding=list(embedding.vector),
        fetched_at=fetched_at,
    )


def _absorb_authors(
    *,
    paper_id: str,
    authors: list[Author] | None,
    author_rows: dict[str, S2AuthorRow],
    paper_authors: dict[tuple[str, str], S2PaperAuthorRow],
) -> None:
    if not authors:
        return
    for position, author in enumerate(authors):
        row = author_row_from(author)
        if row is None:
            continue
        author_rows.setdefault(row.author_id, row)
        key = (paper_id, row.author_id)
        paper_authors.setdefault(
            key,
            S2PaperAuthorRow(paper_id=paper_id, author_id=row.author_id, position=position),
        )


def _absorb_related_chunk(
    *,
    subject_paper_id: str,
    related: Sequence[RelatedPaper],
    kind: str,
    papers: dict[str, S2PaperRow],
    authors: dict[str, S2AuthorRow],
    paper_authors: dict[tuple[str, str], S2PaperAuthorRow],
    references: dict[tuple[str, str], S2ReferenceRow],
    citations: dict[tuple[str, str], S2CitationRow],
) -> None:
    """Map one chunk of related papers into stub/author/edge dicts."""
    for item in related:
        if not item.paper_id or item.paper_id == subject_paper_id:
            continue
        stub = paper_row_from(item)
        if stub is None:
            continue
        papers.setdefault(stub.paper_id, stub)
        _absorb_authors(
            paper_id=stub.paper_id,
            authors=item.authors,
            author_rows=authors,
            paper_authors=paper_authors,
        )
        if kind == "reference":
            references[(subject_paper_id, stub.paper_id)] = S2ReferenceRow(
                paper_id=subject_paper_id,
                referenced_paper_id=stub.paper_id,
            )
        else:
            citations[(subject_paper_id, stub.paper_id)] = S2CitationRow(
                paper_id=subject_paper_id,
                citing_paper_id=stub.paper_id,
            )


def subject_graph_from(
    record: S2Record,
    *,
    fetched_at: datetime | None = None,
) -> S2GraphWrite:
    """Subject ``s2_paper`` + its authors / authorship + optional embedding."""
    if not record.paper_id:
        raise ValueError("S2Record.paper_id is required for S2GraphWrite")

    now = fetched_at if fetched_at is not None else datetime.now(UTC)
    authors: dict[str, S2AuthorRow] = {}
    paper_authors: dict[tuple[str, str], S2PaperAuthorRow] = {}

    subject = paper_row_from(record, fetched_at=now, include_tldr=True)
    assert subject is not None  # paper_id checked above
    _absorb_authors(
        paper_id=subject.paper_id,
        authors=record.authors,
        author_rows=authors,
        paper_authors=paper_authors,
    )
    return S2GraphWrite(
        papers=[subject],
        authors=list(authors.values()),
        paper_authors=list(paper_authors.values()),
        references=[],
        citations=[],
        embedding=embedding_row_from(subject.paper_id, record.embedding, fetched_at=now),
    )


def related_graph_from(
    subject_paper_id: str,
    related: Sequence[RelatedPaper],
    *,
    kind: str,
) -> S2GraphWrite:
    """Stub papers + authors + citation or reference edges for one related chunk."""
    papers: dict[str, S2PaperRow] = {}
    authors: dict[str, S2AuthorRow] = {}
    paper_authors: dict[tuple[str, str], S2PaperAuthorRow] = {}
    references: dict[tuple[str, str], S2ReferenceRow] = {}
    citations: dict[tuple[str, str], S2CitationRow] = {}
    _absorb_related_chunk(
        subject_paper_id=subject_paper_id,
        related=related,
        kind=kind,
        papers=papers,
        authors=authors,
        paper_authors=paper_authors,
        references=references,
        citations=citations,
    )
    return S2GraphWrite(
        papers=list(papers.values()),
        authors=list(authors.values()),
        paper_authors=list(paper_authors.values()),
        references=list(references.values()),
        citations=list(citations.values()),
        embedding=None,
    )


def from_paper_record(
    record: S2Record,
    *,
    fetched_at: datetime | None = None,
) -> S2GraphWrite:
    """Pure mapper: one API S2Record → multi-table upsert payload.

    Subject paper wins on id collisions with citation/reference stubs.
    Authors and edges are deduped by natural key. Papers without paper_id
    are dropped. Embedding is included only when present and dim == 768.

    For production persistence prefer :func:`s2_writes`, which splits this
    payload into subject + ≤1k-edge chunks + outcome.
    """
    if not record.paper_id:
        raise ValueError("S2Record.paper_id is required for S2GraphWrite")

    now = fetched_at if fetched_at is not None else datetime.now(UTC)
    subject = subject_graph_from(record, fetched_at=now)

    papers: dict[str, S2PaperRow] = {p.paper_id: p for p in subject.papers}
    authors: dict[str, S2AuthorRow] = {a.author_id: a for a in subject.authors}
    paper_authors: dict[tuple[str, str], S2PaperAuthorRow] = {
        (pa.paper_id, pa.author_id): pa for pa in subject.paper_authors
    }
    references: dict[tuple[str, str], S2ReferenceRow] = {}
    citations: dict[tuple[str, str], S2CitationRow] = {}

    _absorb_related_chunk(
        subject_paper_id=record.paper_id,
        related=record.references or (),
        kind="reference",
        papers=papers,
        authors=authors,
        paper_authors=paper_authors,
        references=references,
        citations=citations,
    )
    _absorb_related_chunk(
        subject_paper_id=record.paper_id,
        related=record.citations or (),
        kind="citation",
        papers=papers,
        authors=authors,
        paper_authors=paper_authors,
        references=references,
        citations=citations,
    )

    return S2GraphWrite(
        papers=list(papers.values()),
        authors=list(authors.values()),
        paper_authors=list(paper_authors.values()),
        references=list(references.values()),
        citations=list(citations.values()),
        embedding=subject.embedding,
    )


def outcome_row_from(
    seed_paper_id: int,
    record: S2Record,
    *,
    fetched_at: datetime | None = None,
) -> SourceSemanticScholarRow:
    """Build the ``source_semantic_scholar`` success row for one fetched record.

    Keyed by the seed's bigint id; S2's own id lands in ``s2_paper_id``. ``raw``
    stays None — S2Record does not retain the API payload (unlike the arxiv /
    openalex adapters), so there is nothing faithful to store there yet.
    """
    oa = record.open_access_pdf
    return SourceSemanticScholarRow(
        paper_id=seed_paper_id,
        status="succeeded",
        s2_paper_id=record.paper_id,
        corpus_id=record.corpus_id,
        title=record.title,
        external_ids=dict(record.external_ids) if record.external_ids else None,
        open_access_pdf_url=oa.url if oa else None,
        reference_count=record.reference_count,
        citation_count=record.citation_count,
        fields_of_study=record.fields_of_study,
        fetched_at=fetched_at,
    )


def iter_s2_writes(
    seed_paper_id: int,
    record: S2Record,
    *,
    citation_chunks: Iterable[Sequence[RelatedPaper]] | None = None,
    reference_chunks: Iterable[Sequence[RelatedPaper]] | None = None,
    fetched_at: datetime | None = None,
    chunk_size: int = S2_EDGE_UPSERT_CHUNK,
) -> Iterator[Write]:
    """Stream S2 Writes: subject → edge chunks → outcome (outcome last).

    Pass ``citation_chunks`` / ``reference_chunks`` from a live paginator (preferred
    — keeps Attention-scale graphs out of RSS). When omitted, slices
    ``record.citations`` / ``record.references`` by ``chunk_size`` (unit tests).
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if not record.paper_id:
        raise ValueError("S2Record.paper_id is required for s2_writes")

    now = fetched_at if fetched_at is not None else datetime.now(UTC)
    yield subject_graph_from(record, fetched_at=now).to_write()

    def _chunks(
        explicit: Iterable[Sequence[RelatedPaper]] | None,
        related: Sequence[RelatedPaper] | None,
    ) -> Iterator[Sequence[RelatedPaper]]:
        if explicit is not None:
            yield from explicit
            return
        items = list(related or ())
        for i in range(0, len(items), chunk_size):
            yield items[i : i + chunk_size]

    for related in _chunks(citation_chunks, record.citations):
        chunk = related_graph_from(record.paper_id, related, kind="citation").to_write()
        if chunk.groups:
            yield chunk

    for related in _chunks(reference_chunks, record.references):
        chunk = related_graph_from(record.paper_id, related, kind="reference").to_write()
        if chunk.groups:
            yield chunk

    yield Write(groups=[[outcome_row_from(seed_paper_id, record, fetched_at=now)]])


def s2_writes(
    seed_paper_id: int,
    record: S2Record,
    *,
    fetched_at: datetime | None = None,
    chunk_size: int = S2_EDGE_UPSERT_CHUNK,
) -> list[Write]:
    """Materialize :func:`iter_s2_writes` (subject → ≤chunk_size edges → outcome)."""
    return list(
        iter_s2_writes(
            seed_paper_id,
            record,
            fetched_at=fetched_at,
            chunk_size=chunk_size,
        )
    )


def s2_write(
    seed_paper_id: int,
    record: S2Record,
    *,
    fetched_at: datetime | None = None,
) -> Write:
    """Full S2 persistence payload for one fetched paper (single Write).

    Prefer :func:`s2_writes` for production — this joins subject + all edges +
    outcome into one Write and can OOM / statement-timeout on large graphs.
    """
    now = fetched_at if fetched_at is not None else datetime.now(UTC)
    graph = from_paper_record(record, fetched_at=now).to_write()
    outcome = outcome_row_from(seed_paper_id, record, fetched_at=now)
    return Write(groups=[*graph.groups, [outcome]])
