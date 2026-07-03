from __future__ import annotations

import json

import pytest
from pipeline_v2.contracts import (
    ContractValidationError,
    DocsEnrichedMessage,
    deadletter_topic,
    try_extract_identity,
    validate_message,
)


def test_validate_message_accepts_well_formed_body():
    body = json.dumps(
        {
            "doc_id": "d1",
            "corpus": "glr",
            "content_sha256": "sha",
            "stage_fingerprint": "fp",
        }
    ).encode()
    msg = validate_message("docs.enriched", body)
    assert isinstance(msg, DocsEnrichedMessage)
    assert msg.doc_id == "d1"
    assert msg.attempt == 1


def test_validate_message_rejects_invalid_json():
    with pytest.raises(ContractValidationError):
        validate_message("docs.enriched", b"not json{{{")


def test_validate_message_rejects_missing_required_field():
    body = json.dumps({"doc_id": "d1", "corpus": "glr"}).encode()
    with pytest.raises(ContractValidationError):
        validate_message("docs.enriched", body)


def test_validate_message_rejects_unknown_topic():
    body = json.dumps(
        {"doc_id": "d1", "corpus": "glr", "content_sha256": "s", "stage_fingerprint": "f"}
    ).encode()
    with pytest.raises(ContractValidationError):
        validate_message("no.such.topic", body)


def test_try_extract_identity_recovers_partial_fields_from_valid_json():
    body = json.dumps({"doc_id": "d1", "corpus": "glr", "extra": "ignored"}).encode()
    identity = try_extract_identity(body)
    assert identity == {"doc_id": "d1", "corpus": "glr"}


def test_try_extract_identity_returns_none_for_garbage():
    assert try_extract_identity(b"\xff\xfenot json") is None
    assert try_extract_identity(b"not json{{{") is None
    assert try_extract_identity(json.dumps([1, 2, 3]).encode()) is None


def test_deadletter_topic_naming():
    assert deadletter_topic("extract") == "deadletter.extract"
    assert deadletter_topic("embed") == "deadletter.embed"
    assert deadletter_topic("indexer") == "deadletter.indexer"
    with pytest.raises(ValueError):
        deadletter_topic("not-a-stage")
