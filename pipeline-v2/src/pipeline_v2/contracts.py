"""Pydantic message contracts, one per Kafka topic, plus JSON Schema export.

Every stage-to-stage message carries the same identity quadruple used by
``work_state`` (``doc_id`` / ``corpus`` / ``content_sha256`` /
``stage_fingerprint``) so a consumer can always look up (or write) the
matching work-state row without any side-channel. ``attempt`` mirrors the
delivery/redelivery count semantics used elsewhere in the pipeline (starts
at 1). ``payload`` is a free-form dict — stage-specific data lives there so
this module doesn't need to change shape every time a stage's inputs change.

Topic -> model mapping (``TOPIC_MODELS``) is the single source of truth for
both runtime validation (:mod:`pipeline_v2.consumers`) and JSON Schema
export (``schemas/<topic>.json``, kept current by
``tests/test_contracts_schema_export.py`` via a regenerate-and-diff check).

Deadletter topics are named ``deadletter.<stage>`` (one per stage, not one
global topic) so a consumer can subscribe to just the DLQ for the stage it
owns. ``STAGES`` is the closed set of pipeline stage names used to build
those topic names and to name the input/output topics stitched together in
:mod:`pipeline_v2.consumers`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

STAGES: tuple[str, ...] = ("extract", "embed", "indexer")

SCHEMAS_DIR = Path(__file__).parent.parent.parent / "schemas"


class StageMessage(BaseModel):
    """Base envelope produced/consumed by every pipeline stage topic."""

    doc_id: str
    corpus: str
    content_sha256: str
    stage_fingerprint: str
    attempt: int = 1
    payload: dict[str, Any] = Field(default_factory=dict)


class DocsEnrichedMessage(StageMessage):
    """Produced by the extract stage; consumed by the embed stage."""


class ChunksReadyMessage(StageMessage):
    """Produced by the chunking step of the embed stage's input side."""


class ChunksEmbeddedMessage(StageMessage):
    """Produced by the embed stage; consumed by the indexer stage."""


class IndexCommittedMessage(StageMessage):
    """Produced by the indexer stage once a doc/stage is durably indexed."""


class DeadLetterMessage(StageMessage):
    """Envelope for every ``deadletter.<stage>`` topic.

    Wraps the original message's identity plus why it died and how many
    delivery attempts it had accrued. ``original_topic`` records where the
    poisoned/exhausted record came from so a human triaging the DLQ doesn't
    have to guess.
    """

    reason: str
    original_topic: str
    stage: str


TOPIC_MODELS: dict[str, type[StageMessage]] = {
    "docs.enriched": DocsEnrichedMessage,
    "chunks.ready": ChunksReadyMessage,
    "chunks.embedded": ChunksEmbeddedMessage,
    "index.committed": IndexCommittedMessage,
    **{f"deadletter.{stage}": DeadLetterMessage for stage in STAGES},
}


def deadletter_topic(stage: str) -> str:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}: must be one of {STAGES}")
    return f"deadletter.{stage}"


class ContractValidationError(Exception):
    """Raised when a raw message body does not satisfy its topic's contract.

    Callers (the consumer loop) treat this as a poison message: it cannot be
    processed and goes straight to the dead-letter topic regardless of
    delivery count.
    """


def validate_message(topic: str, raw_value: bytes) -> StageMessage:
    """Parse+validate ``raw_value`` against the contract registered for ``topic``.

    Raises :class:`ContractValidationError` (wrapping the JSON/pydantic
    error) on any failure — unknown topic, invalid JSON, or a schema
    mismatch.
    """
    model = TOPIC_MODELS.get(topic)
    if model is None:
        raise ContractValidationError(f"no contract registered for topic {topic!r}")
    try:
        data = json.loads(raw_value)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ContractValidationError(f"invalid JSON on topic {topic!r}: {exc}") from exc
    try:
        return model.model_validate(data)
    except Exception as exc:  # pydantic.ValidationError, plus defensive catch-all
        raise ContractValidationError(f"contract violation on topic {topic!r}: {exc}") from exc


def try_extract_identity(raw_value: bytes) -> dict[str, str] | None:
    """Best-effort partial parse used only to label an unparseable poison message.

    Returns a dict with whatever of doc_id/corpus/content_sha256/
    stage_fingerprint it could find, or None if the payload isn't even valid
    JSON. Never raises.
    """
    try:
        data = json.loads(raw_value)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    return {
        key: str(data[key])
        for key in ("doc_id", "corpus", "content_sha256", "stage_fingerprint")
        if key in data
    }


def export_json_schema(topic: str) -> dict[str, Any]:
    """Return the JSON Schema for ``topic``'s contract model."""
    model = TOPIC_MODELS.get(topic)
    if model is None:
        raise ValueError(f"no contract registered for topic {topic!r}")
    return model.model_json_schema()


def export_all_json_schemas() -> dict[str, dict[str, Any]]:
    return {topic: export_json_schema(topic) for topic in TOPIC_MODELS}


def schema_filename(topic: str) -> Path:
    return SCHEMAS_DIR / f"{topic}.schema.json"


def write_json_schemas(out_dir: Path = SCHEMAS_DIR) -> list[Path]:
    """Regenerate every ``schemas/<topic>.schema.json`` file. Returns paths written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for topic, schema in export_all_json_schemas().items():
        path = out_dir / f"{topic}.schema.json"
        path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(path)
    return written
