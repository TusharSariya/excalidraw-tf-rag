"""Typed adapter models for Semantic Scholar Graph API paper payloads.

Maps camelCase API JSON into snake_case dataclasses. Open-ended maps
(``externalIds``, ``citationStyles``) stay as dicts; everything else is typed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from semanticscholar import Paper as SemanticScholarPaper

#: Top-level keys on a full paper response (Graph API / SDK ``Paper.FIELDS`` shape).
TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "abstract",
        "authors",
        "citationCount",
        "citationStyles",
        "citations",
        "corpusId",
        "embedding",
        "externalIds",
        "fieldsOfStudy",
        "influentialCitationCount",
        "isOpenAccess",
        "journal",
        "openAccessPdf",
        "paperId",
        "publicationDate",
        "publicationTypes",
        "publicationVenue",
        "referenceCount",
        "references",
        "s2FieldsOfStudy",
        "title",
        "tldr",
        "url",
        "venue",
        "year",
    }
)

#: Keys present on citation/reference paper objects (SEARCH_FIELDS-shaped; no nested graphs).
RELATED_PAPER_KEYS: frozenset[str] = frozenset(
    {
        "abstract",
        "authors",
        "citationCount",
        "citationStyles",
        "corpusId",
        "externalIds",
        "fieldsOfStudy",
        "influentialCitationCount",
        "isOpenAccess",
        "journal",
        "openAccessPdf",
        "paperId",
        "publicationDate",
        "publicationTypes",
        "publicationVenue",
        "referenceCount",
        "s2FieldsOfStudy",
        "title",
        "url",
        "venue",
        "year",
    }
)

ExternalIds = dict[str, str | int | list[str]]
CitationStyles = dict[str, str]


@dataclass(frozen=True)
class Author:
    author_id: str | None = None
    name: str | None = None
    url: str | None = None
    affiliations: list[str] | None = None
    homepage: str | None = None
    paper_count: int | None = None
    citation_count: int | None = None
    h_index: int | None = None
    external_ids: ExternalIds | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Author | None:
        if data is None:
            return None
        return cls(
            author_id=data.get("authorId"),
            name=data.get("name"),
            url=data.get("url"),
            affiliations=data.get("affiliations"),
            homepage=data.get("homepage"),
            paper_count=data.get("paperCount"),
            citation_count=data.get("citationCount"),
            h_index=data.get("hIndex"),
            external_ids=data.get("externalIds"),
        )


@dataclass(frozen=True)
class Journal:
    name: str | None = None
    volume: str | None = None
    pages: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Journal | None:
        if data is None:
            return None
        return cls(
            name=data.get("name"),
            volume=data.get("volume"),
            pages=data.get("pages"),
        )


@dataclass(frozen=True)
class OpenAccessPdf:
    url: str | None = None
    status: str | None = None
    license: str | None = None
    disclaimer: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> OpenAccessPdf | None:
        if data is None:
            return None
        return cls(
            url=data.get("url") or None,
            status=data.get("status"),
            license=data.get("license"),
            disclaimer=data.get("disclaimer"),
        )


@dataclass(frozen=True)
class PublicationVenue:
    id: str | None = None
    name: str | None = None
    type: str | None = None
    alternate_names: list[str] | None = None
    issn: str | None = None
    url: str | None = None
    alternate_urls: list[str] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PublicationVenue | None:
        if data is None:
            return None
        return cls(
            id=data.get("id"),
            name=data.get("name"),
            type=data.get("type"),
            alternate_names=data.get("alternate_names"),
            issn=data.get("issn"),
            url=data.get("url"),
            alternate_urls=data.get("alternate_urls"),
        )


@dataclass(frozen=True)
class Tldr:
    model: str | None = None
    text: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Tldr | None:
        if data is None:
            return None
        return cls(model=data.get("model"), text=data.get("text"))


@dataclass(frozen=True)
class Embedding:
    model: str | None = None
    vector: list[float] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Embedding | None:
        if data is None:
            return None
        return cls(model=data.get("model"), vector=data.get("vector"))


@dataclass(frozen=True)
class FieldOfStudy:
    category: str | None = None
    source: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> FieldOfStudy | None:
        if data is None:
            return None
        return cls(category=data.get("category"), source=data.get("source"))


def _authors(data: list[dict[str, Any]] | None) -> list[Author]:
    if not data:
        return []
    return [a for item in data if (a := Author.from_dict(item)) is not None]


def _fields_of_study(data: list[dict[str, Any]] | None) -> list[FieldOfStudy]:
    if not data:
        return []
    return [f for item in data if (f := FieldOfStudy.from_dict(item)) is not None]


@dataclass(frozen=True)
class RelatedPaper:
    """Citation/reference paper — SEARCH_FIELDS shape (no nested citations/references)."""

    paper_id: str | None = None
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
    external_ids: ExternalIds | None = None
    citation_styles: CitationStyles | None = None
    authors: list[Author] | None = None
    journal: Journal | None = None
    open_access_pdf: OpenAccessPdf | None = None
    publication_venue: PublicationVenue | None = None
    s2_fields_of_study: list[FieldOfStudy] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RelatedPaper | None:
        if data is None:
            return None
        unknown = set(data) - RELATED_PAPER_KEYS
        if unknown:
            raise ValueError(f"RelatedPaper: unknown field(s) {sorted(unknown)}")
        return cls(
            paper_id=data.get("paperId"),
            corpus_id=data.get("corpusId"),
            title=data.get("title"),
            abstract=data.get("abstract"),
            venue=data.get("venue"),
            year=data.get("year"),
            url=data.get("url"),
            reference_count=data.get("referenceCount"),
            citation_count=data.get("citationCount"),
            influential_citation_count=data.get("influentialCitationCount"),
            is_open_access=data.get("isOpenAccess"),
            publication_date=data.get("publicationDate"),
            publication_types=data.get("publicationTypes"),
            fields_of_study=data.get("fieldsOfStudy"),
            external_ids=data.get("externalIds"),
            citation_styles=data.get("citationStyles"),
            authors=_authors(data.get("authors")),
            journal=Journal.from_dict(data.get("journal")),
            open_access_pdf=OpenAccessPdf.from_dict(data.get("openAccessPdf")),
            publication_venue=PublicationVenue.from_dict(data.get("publicationVenue")),
            s2_fields_of_study=_fields_of_study(data.get("s2FieldsOfStudy")),
        )


@dataclass(frozen=True)
class PaperRecord:
    """Full typed Semantic Scholar paper record (adapter over API / SDK payload)."""

    paper_id: str | None = None
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
    external_ids: ExternalIds | None = None
    citation_styles: CitationStyles | None = None
    authors: list[Author] | None = None
    journal: Journal | None = None
    open_access_pdf: OpenAccessPdf | None = None
    publication_venue: PublicationVenue | None = None
    s2_fields_of_study: list[FieldOfStudy] | None = None
    tldr: Tldr | None = None
    embedding: Embedding | None = None
    citations: list[RelatedPaper] | None = None
    references: list[RelatedPaper] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaperRecord:
        unknown = set(data) - TOP_LEVEL_KEYS
        if unknown:
            raise ValueError(f"PaperRecord: unknown top-level field(s) {sorted(unknown)}")

        citations_raw = data.get("citations")
        references_raw = data.get("references")
        return cls(
            paper_id=data.get("paperId"),
            corpus_id=data.get("corpusId"),
            title=data.get("title"),
            abstract=data.get("abstract"),
            venue=data.get("venue"),
            year=data.get("year"),
            url=data.get("url"),
            reference_count=data.get("referenceCount"),
            citation_count=data.get("citationCount"),
            influential_citation_count=data.get("influentialCitationCount"),
            is_open_access=data.get("isOpenAccess"),
            publication_date=data.get("publicationDate"),
            publication_types=data.get("publicationTypes"),
            fields_of_study=data.get("fieldsOfStudy"),
            external_ids=data.get("externalIds"),
            citation_styles=data.get("citationStyles"),
            authors=_authors(data.get("authors")),
            journal=Journal.from_dict(data.get("journal")),
            open_access_pdf=OpenAccessPdf.from_dict(data.get("openAccessPdf")),
            publication_venue=PublicationVenue.from_dict(data.get("publicationVenue")),
            s2_fields_of_study=_fields_of_study(data.get("s2FieldsOfStudy")),
            tldr=Tldr.from_dict(data.get("tldr")),
            embedding=Embedding.from_dict(data.get("embedding")),
            citations=(
                [r for item in citations_raw if (r := RelatedPaper.from_dict(item))]
                if citations_raw is not None
                else None
            ),
            references=(
                [r for item in references_raw if (r := RelatedPaper.from_dict(item))]
                if references_raw is not None
                else None
            ),
        )

    @classmethod
    def from_sdk(cls, paper: SemanticScholarPaper) -> PaperRecord:
        return cls.from_dict(dict(paper.raw_data))
