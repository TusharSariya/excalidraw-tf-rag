"""Typed projection of an Unpaywall v2 response.

Unpaywall's ``GET /v2/{doi}`` returns a bare JSON object — no schema, no
validation. This module is the schema it declines to give: frozen dataclasses
over the fields harvest actually reads, each with a tolerant ``from_dict``.

Design choices, mirroring ``OpenAlexWork`` / ``ArxivRecord`` / ``S2Record``:

* **Tolerant, not strict.** Keep only the fields we know and ignore the rest —
  Unpaywall adds fields over time and a harvester should not break when it does.
  Nothing is lost regardless: the whole response is retained on ``raw`` for the
  ``source_unpaywall.raw`` jsonb column.
* **Everything optional.** Unpaywall omits/nulls fields freely
  (``best_oa_location`` is ``None`` for a closed-access DOI, ``url_for_pdf`` is
  ``None`` when only a landing page is known, …), so every field defaults and
  nested access is guarded.

Field names/nesting follow the documented Unpaywall v2 schema; the whole payload
is kept on ``raw`` so the projection can stay small.
See https://unpaywall.org/data-formats for the full catalog.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OaLocation:
    """One open-access copy — ``best_oa_location`` or an ``oa_locations`` item."""

    url: str | None = None
    url_for_pdf: str | None = None
    url_for_landing_page: str | None = None
    host_type: str | None = None
    version: str | None = None
    license: str | None = None
    evidence: str | None = None
    is_best: bool | None = None
    pmh_id: str | None = None
    endpoint_id: str | None = None
    repository_institution: str | None = None
    updated: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> OaLocation | None:
        if not data:
            return None
        return cls(
            url=data.get("url"),
            url_for_pdf=data.get("url_for_pdf"),
            url_for_landing_page=data.get("url_for_landing_page"),
            host_type=data.get("host_type"),
            version=data.get("version"),
            license=data.get("license"),
            evidence=data.get("evidence"),
            is_best=data.get("is_best"),
            pmh_id=data.get("pmh_id"),
            endpoint_id=data.get("endpoint_id"),
            repository_institution=data.get("repository_institution"),
            updated=data.get("updated"),
        )


def _list(factory, data: list[dict[str, Any]] | None) -> list:
    """Map a list of dicts through a ``from_dict``, dropping anything that parses to None."""
    return [item for item in map(factory, data or []) if item is not None]


@dataclass(frozen=True)
class UnpaywallRecord:
    """A typed Unpaywall v2 response — the subset harvest reads, plus ``raw``.

    ``from_result`` takes the plain dict the API hands back for one DOI. Keep the
    whole dict on ``raw`` so ``source_unpaywall.raw`` stays a faithful copy even
    as this projection stays small.
    """

    doi: str
    is_oa: bool | None = None
    oa_status: str | None = None
    title: str | None = None
    genre: str | None = None
    published_date: str | None = None
    year: int | None = None
    publisher: str | None = None
    journal_name: str | None = None
    journal_issn_l: str | None = None
    is_paratext: bool | None = None
    has_repository_copy: bool | None = None
    best_oa_location: OaLocation | None = None
    oa_locations: list[OaLocation] = field(default_factory=list)
    updated: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def url_for_pdf(self) -> str | None:
        """The primary open-access PDF URL, if Unpaywall found one."""
        return self.best_oa_location.url_for_pdf if self.best_oa_location else None

    @property
    def best_oa_url(self) -> str | None:
        """The best open-access URL (landing page or PDF), if any."""
        return self.best_oa_location.url if self.best_oa_location else None

    @property
    def best_oa_host_type(self) -> str | None:
        return self.best_oa_location.host_type if self.best_oa_location else None

    @property
    def best_oa_version(self) -> str | None:
        return self.best_oa_location.version if self.best_oa_location else None

    @property
    def best_oa_license(self) -> str | None:
        return self.best_oa_location.license if self.best_oa_location else None

    @classmethod
    def from_result(cls, result: dict[str, Any]) -> UnpaywallRecord:
        return cls(
            doi=result.get("doi") or "",
            is_oa=result.get("is_oa"),
            oa_status=result.get("oa_status"),
            title=result.get("title"),
            genre=result.get("genre"),
            published_date=result.get("published_date"),
            year=result.get("year"),
            publisher=result.get("publisher"),
            journal_name=result.get("journal_name"),
            journal_issn_l=result.get("journal_issn_l"),
            is_paratext=result.get("is_paratext"),
            has_repository_copy=result.get("has_repository_copy"),
            best_oa_location=OaLocation.from_dict(result.get("best_oa_location")),
            oa_locations=_list(OaLocation.from_dict, result.get("oa_locations")),
            updated=result.get("updated"),
            raw=dict(result),
        )
