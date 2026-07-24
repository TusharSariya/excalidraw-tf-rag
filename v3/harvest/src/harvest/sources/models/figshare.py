"""Typed projection of a figshare article record.

figshare's ``GET /v2/articles/{id}`` returns a bare JSON object — no schema, no
validation. This module is the schema it declines to give: frozen dataclasses
over the fields harvest actually reads, each with a tolerant ``from_dict``.

Design choices, mirroring ``OpenAlexWork`` / ``UnpaywallRecord`` / ``S2Record``:

* **Tolerant, not strict.** Keep only the fields we know and ignore the rest —
  figshare adds fields over time and a harvester should not break when it does.
  Nothing is lost regardless: the whole response is retained on ``raw`` for the
  ``source_figshare.raw`` jsonb column.
* **Everything optional.** figshare omits/nulls fields freely (``files`` is empty
  for a metadata-only record, ``mimetype`` is null, ``orcid_id`` is an empty
  string when absent, …), so every field defaults and nested access is guarded.

A crucial identity note: figshare mints its OWN ``doi`` for each deposit (e.g.
``10.1371/journal.pone.0098679.g001``), which is NOT the seed's publication DOI.
The publication DOI harvest searches by and keys on lives in ``resource_doi``.

Field names/nesting follow a live ``GET /articles/{id}`` response; the whole
payload is kept on ``raw`` so the projection can stay small.
See https://docs.figshare.com/#public_article for the full catalog.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FigshareFile:
    """One deposit file — an item of the article's ``files`` array."""

    id: int | None = None
    name: str | None = None
    size: int | None = None
    download_url: str | None = None
    supplied_md5: str | None = None
    computed_md5: str | None = None
    mimetype: str | None = None
    is_link_only: bool | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> FigshareFile | None:
        if not data:
            return None
        return cls(
            id=data.get("id"),
            name=data.get("name"),
            size=data.get("size"),
            download_url=data.get("download_url"),
            supplied_md5=data.get("supplied_md5"),
            computed_md5=data.get("computed_md5"),
            mimetype=data.get("mimetype"),
            is_link_only=data.get("is_link_only"),
        )


@dataclass(frozen=True)
class FigshareAuthor:
    """One entry of the article's ``authors`` array."""

    id: int | None = None
    full_name: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    orcid_id: str | None = None
    url_name: str | None = None
    is_active: bool | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> FigshareAuthor | None:
        if not data:
            return None
        return cls(
            id=data.get("id"),
            full_name=data.get("full_name"),
            first_name=data.get("first_name"),
            last_name=data.get("last_name"),
            # empty string when absent — normalise to None so downstream can test truthiness.
            orcid_id=data.get("orcid_id") or None,
            url_name=data.get("url_name"),
            is_active=data.get("is_active"),
        )


@dataclass(frozen=True)
class FigshareLicense:
    """The article's ``license`` block."""

    value: int | None = None
    name: str | None = None
    url: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> FigshareLicense | None:
        if not data:
            return None
        return cls(
            value=data.get("value"),
            name=data.get("name"),
            url=data.get("url"),
        )


@dataclass(frozen=True)
class FigshareCategory:
    """One entry of the article's ``categories`` array (figshare subject taxonomy)."""

    id: int | None = None
    title: str | None = None
    parent_id: int | None = None
    source_id: str | None = None
    taxonomy_id: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> FigshareCategory | None:
        if not data:
            return None
        return cls(
            id=data.get("id"),
            title=data.get("title"),
            parent_id=data.get("parent_id"),
            source_id=data.get("source_id"),
            taxonomy_id=data.get("taxonomy_id"),
        )


def _list(factory, data: list[dict[str, Any]] | None) -> list:
    """Map a list of dicts through a ``from_dict``, dropping anything that parses to None."""
    return [item for item in map(factory, data or []) if item is not None]


@dataclass(frozen=True)
class FigshareArticle:
    """A typed figshare article — the subset harvest reads, plus ``raw``.

    ``from_result`` takes the plain dict the ``GET /articles/{id}`` endpoint hands
    back. Keep the whole dict on ``raw`` so ``source_figshare.raw`` stays a
    faithful copy even as this projection stays small.
    """

    id: int
    title: str | None = None
    doi: str | None = None  # figshare-minted deposit DOI, NOT the seed publication DOI
    resource_doi: str | None = None  # the associated publication DOI (the seed doi)
    resource_title: str | None = None
    description: str | None = None
    figshare_url: str | None = None
    url_public_html: str | None = None
    published_date: str | None = None
    defined_type_name: str | None = None
    version: int | None = None
    files: list[FigshareFile] = field(default_factory=list)
    authors: list[FigshareAuthor] = field(default_factory=list)
    license: FigshareLicense | None = None
    tags: list[str] = field(default_factory=list)
    categories: list[FigshareCategory] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def download_url(self) -> str | None:
        """The first deposit file's direct download URL, if any."""
        for f in self.files:
            if f.download_url:
                return f.download_url
        return None

    @property
    def license_name(self) -> str | None:
        return self.license.name if self.license else None

    @classmethod
    def from_result(cls, result: dict[str, Any]) -> FigshareArticle:
        return cls(
            id=result.get("id"),
            title=result.get("title"),
            doi=result.get("doi"),
            resource_doi=result.get("resource_doi"),
            resource_title=result.get("resource_title"),
            description=result.get("description"),
            figshare_url=result.get("figshare_url"),
            url_public_html=result.get("url_public_html"),
            published_date=result.get("published_date"),
            defined_type_name=result.get("defined_type_name"),
            version=result.get("version"),
            files=_list(FigshareFile.from_dict, result.get("files")),
            authors=_list(FigshareAuthor.from_dict, result.get("authors")),
            license=FigshareLicense.from_dict(result.get("license")),
            tags=list(result.get("tags") or []),
            categories=_list(FigshareCategory.from_dict, result.get("categories")),
            raw=dict(result),
        )
