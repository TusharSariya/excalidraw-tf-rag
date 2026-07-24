"""Table-shaped DTO for public.source_osf + OsfPreprint -> Write mapper.

Mirrors ``persistence.models.unpaywall``: one row group, keyed by the seed's
bigint ``paper_id``. The typed columns are the queryable subset; the whole OSF
JSON:API response is kept on ``raw``, with the ``subjects`` taxonomy broken out
to its own jsonb column for convenience.

``OsfPreprint`` is imported only for typing — the mapper reads its attributes and
never constructs one, so persistence keeps no runtime dependency on ``sources``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ..base import TableRow, Write

if TYPE_CHECKING:
    from ...sources.models.osf import OsfPreprint


class SourceOsfRow(TableRow):
    """Insert/upsert shape for public.source_osf (per-seed outcome).

    Keyed by the seed's ``paper_id``. ``subjects``, ``license_record`` and
    ``raw`` are jsonb; the scalar columns are the promoted, queryable OSF fields.
    """

    __table__ = "source_osf"
    __conflict__ = ("paper_id",)

    paper_id: int
    status: str = "succeeded"
    osf_id: str | None = None
    doi: str | None = None
    preprint_doi: str | None = None
    title: str | None = None
    description: str | None = None
    date_published: str | None = None
    date_modified: str | None = None
    is_published: bool | None = None
    reviews_state: str | None = None
    provider_id: str | None = None
    node_id: str | None = None
    html_url: str | None = None
    primary_file_id: str | None = None
    primary_file_url: str | None = None
    files_url: str | None = None
    tags: list[str] | None = None
    subjects: list[Any] | None = None
    license_record: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None
    error: str | None = None
    fetched_at: datetime | None = None


def osf_write(
    seed_paper_id: int,
    rec: OsfPreprint,
    *,
    fetched_at: datetime | None = None,
) -> Write:
    """Build the ``source_osf`` success ``Write`` for one fetched ``OsfPreprint``.

    A single row group keyed by the seed's ``paper_id`` — OSF has no citation
    graph, so there is nothing to fan out (contrast ``s2_write``).
    """
    now = fetched_at if fetched_at is not None else datetime.now(UTC)
    # subjects/license_record already live in `raw`; break them out for querying,
    # sourced from raw so the jsonb columns are faithful copies of the payload.
    attributes = rec.raw.get("data") if isinstance(rec.raw.get("data"), dict) else {}
    attributes = attributes.get("attributes") if isinstance(attributes, dict) else {}
    if not isinstance(attributes, dict):
        attributes = {}
    row = SourceOsfRow(
        paper_id=seed_paper_id,
        status="succeeded",
        osf_id=rec.osf_id or None,
        doi=rec.doi,
        preprint_doi=rec.preprint_doi,
        title=rec.title or None,
        description=rec.description,
        date_published=rec.date_published,
        date_modified=rec.date_modified,
        is_published=rec.is_published,
        reviews_state=rec.reviews_state,
        provider_id=rec.provider_id,
        node_id=rec.node_id,
        html_url=rec.html_url,
        primary_file_id=rec.primary_file_id,
        primary_file_url=rec.primary_file_url,
        files_url=rec.files_url,
        tags=rec.tags or None,
        subjects=attributes.get("subjects"),
        license_record=attributes.get("license_record"),
        raw=rec.raw,
        fetched_at=now,
    )
    return Write(groups=[[row]])
