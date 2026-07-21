"""Table-shaped DTO for public.source_openalex + OpenAlexWork -> Write mapper.

Mirrors ``persistence.models.arxiv``: one row group, keyed by the seed's bigint
``paper_id``. The typed columns are the queryable subset (promoted in migration
``20260720120000``); the whole Work is kept on ``raw``, with its ``authorships``
array broken out to the ``authorships`` jsonb column for convenience.

``OpenAlexWork`` is imported only for typing — the mapper reads its attributes and
never constructs one, so persistence keeps no runtime dependency on ``sources``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ..base import TableRow, Write

if TYPE_CHECKING:
    from ...sources.models.openalex import OpenAlexWork


class SourceOpenAlexRow(TableRow):
    """Insert/upsert shape for public.source_openalex (per-seed outcome).

    Keyed by the seed's ``paper_id``. ``authorships`` and ``raw`` are jsonb; the
    scalar columns are the promoted, queryable OpenAlex fields.
    """

    __table__ = "source_openalex"
    __conflict__ = ("paper_id",)

    paper_id: int
    status: str = "succeeded"
    openalex_id: str | None = None
    doi: str | None = None
    title: str | None = None
    publication_date: str | None = None
    publication_year: int | None = None
    language: str | None = None
    type: str | None = None
    best_oa_pdf_url: str | None = None
    is_oa: bool | None = None
    oa_status: str | None = None
    cited_by_count: int | None = None
    referenced_works_count: int | None = None
    is_retracted: bool | None = None
    authorships: list[dict[str, Any]] | None = None
    raw: dict[str, Any] | None = None
    error: str | None = None
    fetched_at: datetime | None = None


def openalex_write(
    seed_paper_id: int,
    work: OpenAlexWork,
    *,
    fetched_at: datetime | None = None,
) -> Write:
    """Build the ``source_openalex`` success ``Write`` for one fetched ``OpenAlexWork``.

    A single row group keyed by the seed's ``paper_id`` — OpenAlex has a citation
    graph (``referenced_works``), but it is kept in ``raw`` rather than fanned into
    per-openalex graph tables, so there is nothing to fan out here (contrast ``s2_write``).
    """
    now = fetched_at if fetched_at is not None else datetime.now(UTC)
    oa = work.open_access
    row = SourceOpenAlexRow(
        paper_id=seed_paper_id,
        status="succeeded",
        openalex_id=work.openalex_id or None,
        doi=work.doi,
        title=work.title or None,
        publication_date=work.publication_date,
        publication_year=work.publication_year,
        language=work.language,
        type=work.type,
        best_oa_pdf_url=work.best_oa_pdf_url,
        is_oa=oa.is_oa if oa else None,
        oa_status=oa.oa_status if oa else None,
        cited_by_count=work.cited_by_count,
        referenced_works_count=work.referenced_works_count,
        is_retracted=work.is_retracted,
        # authorships already lives in `raw`; break it out for convenient querying.
        authorships=work.raw.get("authorships"),
        raw=work.raw,
        fetched_at=now,
    )
    return Write(groups=[[row]])
