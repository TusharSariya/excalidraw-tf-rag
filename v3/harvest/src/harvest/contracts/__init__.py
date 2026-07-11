"""The seed -> message contract, shared by the harvest clients.

`seed_message.v1.json` is the canonical wire schema — the *same* schema the
database enforces inside `public.publish_seed` (via `pg_jsonschema`; see
`v3/supabase/migrations/20260711040000_enforce_seed_message_schema.sql`). This
module loads that file and exposes:

- `SCHEMA`  — the raw JSON Schema (a dict), for interop / validation / docs.
- `SeedMessage` — a typed pydantic view a consumer parses incoming pgmq bodies
  into (`extra="forbid"` mirrors the schema's `additionalProperties: false`).
- `parse(message)` — validate + parse a raw message body, raising on violation.

The message is produced in-database, so Python is the *consumer* of this
contract: the worker calls `parse(rec["message"])` and reads `.paper_id`.
"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

#: Current envelope version. A fatter/self-contained payload would ship as v2.
SCHEMA_VERSION = 1

#: The canonical v1 JSON Schema, loaded from the packaged file.
SCHEMA: dict[str, Any] = json.loads(
    files(__package__).joinpath("seed_message.v1.json").read_text(encoding="utf-8")
)


class SeedMessage(BaseModel):
    """A message published to a `harvest_*` queue (schema v1)."""

    model_config = ConfigDict(extra="forbid")

    v: Literal[1] = SCHEMA_VERSION
    paper_id: int


def parse(message: dict[str, Any]) -> SeedMessage:
    """Validate a raw pgmq message body against the v1 contract and return it typed.

    Raises `pydantic.ValidationError` if the body doesn't match (unknown field,
    wrong version, missing/!int paper_id).
    """
    return SeedMessage.model_validate(message)
