"""Table-shaped DTO for public.source_unpaywall + UnpaywallRecord -> Write mapper.

Mirrors ``persistence.models.openalex``: one row group, keyed by the seed's
bigint ``paper_id``. The typed columns are the queryable subset; the whole
Unpaywall response is kept on ``raw``, with its ``oa_locations`` array broken
out to the ``oa_locations`` jsonb column for convenience.

``UnpaywallRecord`` is imported only for typing — the mapper reads its attributes
and never constructs one, so persistence keeps no runtime dependency on ``sources``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ..base import TableRow, Write

if TYPE_CHECKING:
    from ...sources.models.unpaywall import UnpaywallRecord


class SourceUnpaywallRow(TableRow):
    """Insert/upsert shape for public.source_unpaywall (per-seed outcome).

    Keyed by the seed's ``paper_id``. ``oa_locations`` and ``raw`` are jsonb; the
    scalar columns are the promoted, queryable Unpaywall fields.
    """

    __table__ = "source_unpaywall"
    __conflict__ = ("paper_id",)

    paper_id: int
    status: str = "succeeded"
    doi: str | None = None
    is_oa: bool | None = None
    oa_status: str | None = None
    title: str | None = None
    genre: str | None = None
    published_date: str | None = None
    year: int | None = None
    publisher: str | None = None
    journal_name: str | None = None
    url_for_pdf: str | None = None
    best_oa_url: str | None = None
    best_oa_host_type: str | None = None
    best_oa_version: str | None = None
    best_oa_license: str | None = None
    oa_locations: list[dict[str, Any]] | None = None
    raw: dict[str, Any] | None = None
    error: str | None = None
    fetched_at: datetime | None = None


def unpaywall_write(
    seed_paper_id: int,
    rec: UnpaywallRecord,
    *,
    fetched_at: datetime | None = None,
) -> Write:
    """Build the ``source_unpaywall`` success ``Write`` for one fetched ``UnpaywallRecord``.

    A single row group keyed by the seed's ``paper_id`` — Unpaywall has no
    citation graph, so there is nothing to fan out (contrast ``s2_write``).
    """
    now = fetched_at if fetched_at is not None else datetime.now(UTC)
    row = SourceUnpaywallRow(
        paper_id=seed_paper_id,
        status="succeeded",
        doi=rec.doi or None,
        is_oa=rec.is_oa,
        oa_status=rec.oa_status,
        title=rec.title,
        genre=rec.genre,
        published_date=rec.published_date,
        year=rec.year,
        publisher=rec.publisher,
        journal_name=rec.journal_name,
        url_for_pdf=rec.url_for_pdf,
        best_oa_url=rec.best_oa_url,
        best_oa_host_type=rec.best_oa_host_type,
        best_oa_version=rec.best_oa_version,
        best_oa_license=rec.best_oa_license,
        # oa_locations already lives in `raw`; break it out for convenient querying.
        oa_locations=rec.raw.get("oa_locations"),
        raw=rec.raw,
        fetched_at=now,
    )
    return Write(groups=[[row]])
