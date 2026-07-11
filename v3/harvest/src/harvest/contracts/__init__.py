"""The seed -> message contract, shared by the harvest clients.

`seed_message.v1.json` is the canonical wire schema and the single source of
truth. Two things derive from it:

- `_seed_message_v1.py` — the `SeedMessage` pydantic model, *generated* from the
  JSON Schema (do not hand-edit; run `scripts/regenerate-contracts.sh`).
- `public.seed_message_schema()` in the database — an in-SQL copy the migration
  carries, enforced inside `public.publish_seed` (see
  `v3/supabase/migrations/20260711040000_enforce_seed_message_schema.sql`).

**Tolerant reader.** The schema is an open content model (no
`additionalProperties: false`), so this contract distinguishes additive from
breaking change by *how strictly it reacts*:

- **Additive** (an unknown extra field) -> `parse()` logs a warning (you're
  behind the producer) and ignores it. A new producer never breaks an
  un-upgraded consumer, so additive changes need no synchronized deploy.
- **Subtractive / modified** (a missing required field, a wrong type, or an
  unrecognized `v`) -> `parse()` raises `pydantic.ValidationError`. These are
  the changes that actually corrupt processing, so they fail hard.

The message is produced in-database, so Python is the *consumer* of this
contract: the worker calls `parse(rec["message"])` and reads `.paper_id`.
"""

from __future__ import annotations

import json
import logging
from importlib.resources import files
from typing import Any

from ._seed_message_v1 import SeedMessage

log = logging.getLogger("harvest.contracts")

#: Current envelope version. A fatter/self-contained payload would ship as v2.
SCHEMA_VERSION = 1

#: The canonical v1 JSON Schema, loaded from the packaged file.
SCHEMA: dict[str, Any] = json.loads(
    files(__package__).joinpath("seed_message.v1.json").read_text(encoding="utf-8")
)

#: The fields this consumer understands; anything else is additive drift.
_KNOWN_FIELDS = frozenset(SeedMessage.model_fields)


def parse(message: dict[str, Any]) -> SeedMessage:
    """Validate a raw pgmq message body against the v1 contract and return it typed.

    Tolerant reader: unknown fields (additive drift) are logged and ignored;
    a missing required field, a wrong type, or an unrecognized version raises
    `pydantic.ValidationError`.
    """
    unknown = message.keys() - _KNOWN_FIELDS
    if unknown:
        log.warning(
            "seed message has unhandled field(s) %s (v=%s) — this consumer may be "
            "behind the producer; ignoring them",
            sorted(unknown),
            message.get("v"),
        )
    return SeedMessage.model_validate(message)


__all__ = ["SCHEMA", "SCHEMA_VERSION", "SeedMessage", "parse"]
