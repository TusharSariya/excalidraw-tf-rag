from dataclasses import dataclass
from typing import Protocol


@dataclass
class Author:
    name: str
    author_id: str | None = None


@dataclass
class Paper:
    title: str
    arxiv_id: str | None = None
    doi: str | None = None
    openalex_id: str | None = None
    pdf_url: str | None = None


class PaperSource(Protocol):
    def get_paper(self, identifier: str) -> Paper | None: ...
    def get_authors(self, identifier: str) -> list[Author]: ...
    def get_references(self, identifier: str) -> list[Paper]: ...
    def get_citations(self, identifier: str) -> list[Paper]: ...
    def get_pdf_url(self, identifier: str) -> str | None: ...
