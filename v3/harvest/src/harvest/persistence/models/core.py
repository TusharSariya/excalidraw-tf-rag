"""Table-shaped DTO for public.source_core + CoreWork -> Write mapper.

Mirrors ``persistence.models.openalex``: one row group, keyed by the seed's bigint
``paper_id``. The typed columns are the queryable subset (the OA fulltext URL /
download id being the point of the source); the whole CORE work is kept on ``raw``.

``CoreWork`` is imported only for typing — the mapper reads its attributes and
never constructs one, so persistence keeps no runtime dependency on ``sources``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ..base import TableRow, Write

if TYPE_CHECKING:
    from ...sources.models.core import CoreWork


class SourceCoreRow(TableRow):
    """Insert/upsert shape for public.source_core (per-seed outcome).

    Keyed by the seed's ``paper_id``. ``data_providers`` and ``raw`` are jsonb; the
    scalar columns are the promoted, queryable CORE fields — ``download_url`` /
    ``full_text_identifier`` / ``core_id`` being the OA fulltext URL / download id.
    """

    __table__ = "source_core"
    __conflict__ = ("paper_id",)

    paper_id: int
    status: str = "succeeded"
    core_id: str | None = None
    doi: str | None = None
    title: str | None = None
    download_url: str | None = None
    full_text_identifier: str | None = None
    year_published: int | None = None
    published_date: str | None = None
    publisher: str | None = None
    language: str | None = None
    document_type: str | None = None
    data_providers: list[dict[str, Any]] | None = None
    raw: dict[str, Any] | None = None
    error: str | None = None
    fetched_at: datetime | None = None


def core_write(
    seed_paper_id: int,
    work: CoreWork,
    *,
    fetched_at: datetime | None = None,
) -> Write:
    """Build the ``source_core`` success ``Write`` for one fetched ``CoreWork``.

    A single row group keyed by the seed's ``paper_id`` — CORE exposes no citation
    graph, so there is nothing to fan out (contrast ``s2_write``).
    """
    now = fetched_at if fetched_at is not None else datetime.now(UTC)
    lang = work.language
    row = SourceCoreRow(
        paper_id=seed_paper_id,
        status="succeeded",
        core_id=work.core_id,
        doi=work.doi,
        title=work.title or None,
        download_url=work.download_url,
        full_text_identifier=work.full_text_identifier,
        year_published=work.year_published,
        published_date=work.published_date,
        publisher=work.publisher,
        language=lang.code if lang else None,
        document_type=work.document_type,
        # data_providers already lives in `raw`; break it out for convenient querying.
        data_providers=work.raw.get("dataProviders"),
        raw=work.raw,
        fetched_at=now,
    )
    return Write(groups=[[row]])
