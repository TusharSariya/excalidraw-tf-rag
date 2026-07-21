from typing import Protocol, TypeVar

P = TypeVar("P")
A = TypeVar("A")


class PaperSource(Protocol):
    def get_paper(self, identifier: str) -> P | None: ...
