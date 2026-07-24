from typing import Protocol, TypeVar

P = TypeVar("P")


class PaperSource(Protocol):
    def get_papers(self, identifiers: list[str]) -> dict[str, P | None]: ...
