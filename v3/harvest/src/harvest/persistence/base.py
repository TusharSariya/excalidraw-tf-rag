"""The persistence contract shared by every backend.

A source-mapper turns one API record into a ``Write`` — an ordered set of
homogeneous row-groups. Each row is a ``TableRow`` that carries its own table
name and conflict key, so a backend can upsert any source's rows without ever
naming a source. That collapses the (source × backend) matrix to source + backend.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    # Runtime import would cycle: base -> models -> models.s2 -> base. The
    # annotations are lazy (`from __future__ import annotations`), so a
    # type-checking-only import is all we need.
    from .models import Seed


class TableRow(BaseModel):
    """One upsertable row. Subclasses set the table name and conflict key.

    ``__table__``/``__conflict__`` are ``ClassVar`` (per-class metadata, not
    per-row fields), so pydantic keeps them out of the serialized payload while
    the generic writer reads them to route and upsert the row.
    """

    model_config = ConfigDict(extra="forbid")

    __table__: ClassVar[str]
    __conflict__: ClassVar[tuple[str, ...]]


@dataclass(frozen=True)
class Write:
    """An upsert payload: homogeneous row-groups, parents before children.

    Each group is all-one-table (the writer reads ``group[0].__table__``).
    Group order is the FK order — a group is only upserted after the groups it
    references. A plain dataclass, not a pydantic model, so the polymorphic
    ``TableRow`` subclasses pass through as-is rather than being re-validated
    down to the base type.
    """

    groups: Sequence[Sequence[TableRow]]


class Persistence(Protocol):
    """What every backend must provide.

    Satisfied structurally — an implementation matches this by shape and need
    not inherit it.
    """

    def load_seed(self, paper_id: int) -> Seed: ...
    def save_seed(self, seed: Seed) -> int: ...
    def write(self, w: Write) -> None: ...
    def mark_not_found(self, source: str, paper_id: int) -> None: ...
    def mark_error(self, source: str, paper_id: int, error: str) -> None: ...
