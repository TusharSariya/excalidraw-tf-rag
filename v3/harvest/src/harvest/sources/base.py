from dataclasses import dataclass
from typing import Protocol, TypeVar

P = TypeVar("P")
A = TypeVar("A")


@dataclass
class Author:
    """A source's projected author: a display name, plus a source id if one exists."""

    name: str
    author_id: str | None = None


@dataclass
class Paper:
    """A source's minimal projected paper: a title plus whatever identities and
    PDF URL the source could resolve. Each source fills in only what it has."""

    title: str
    arxiv_id: str | None = None
    doi: str | None = None
    openalex_id: str | None = None
    pdf_url: str | None = None


class PaperSource(Protocol):
    def get_paper(self, identifier: str) -> P | None: ...
