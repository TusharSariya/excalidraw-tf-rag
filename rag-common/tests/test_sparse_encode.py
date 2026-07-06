"""Offline tests for the shared learned-sparse encoders.

No model weights are loaded here: we assert the encoders are lazy (construction
touches no backend), the Protocol accepts a fake deterministic encoder, and the
resolver dispatches to the right class with the right identity metadata.
"""

from __future__ import annotations

from rag_common.sparse_encode import (
    BGE_M3_MODEL_ID,
    SPLADE_V3_MODEL_ID,
    BGEM3SparseEncoder,
    SparseEncoder,
    SpladeV3Encoder,
    resolve_sparse_encoder,
)


class FakeSparseEncoder:
    """Deterministic hash-based token bags — no model, usable in any offline test."""

    model_id = "fake/sparse"
    version = "fake-1"

    def encode_documents(self, texts: list[str]) -> list[dict[int, float]]:
        return [self.encode_query(t) for t in texts]

    def encode_query(self, text: str) -> dict[int, float]:
        bag: dict[int, float] = {}
        for token in text.lower().split():
            tid = (hash(token) & 0xFFFF) or 1
            bag[tid] = bag.get(tid, 0.0) + 1.0
        return bag


def test_fake_encoder_satisfies_protocol() -> None:
    assert isinstance(FakeSparseEncoder(), SparseEncoder)


def test_fake_encoder_is_deterministic() -> None:
    enc = FakeSparseEncoder()
    assert enc.encode_query("stress majorization") == enc.encode_query("stress majorization")
    docs = enc.encode_documents(["a b", "a b"])
    assert docs[0] == docs[1]


def test_bgem3_is_default_and_lazy() -> None:
    enc = resolve_sparse_encoder()
    assert isinstance(enc, BGEM3SparseEncoder)
    assert enc.model_id == BGE_M3_MODEL_ID
    assert enc.version == "bgem3-1"
    # Lazy: no model weights loaded just by constructing / reading identity.
    assert enc._model is None


def test_splade_resolver_and_identity() -> None:
    enc = resolve_sparse_encoder(encoder="splade")
    assert isinstance(enc, SpladeV3Encoder)
    assert enc.model_id == SPLADE_V3_MODEL_ID
    assert enc.version == "splade-1"
    assert enc._model is None


def test_resolver_honours_model_override_and_version() -> None:
    enc = resolve_sparse_encoder(encoder="bgem3", model_id="local/ckpt", version="7")
    assert enc.model_id == "local/ckpt"
    assert enc.version == "bgem3-7"


def test_resolver_rejects_unknown() -> None:
    try:
        resolve_sparse_encoder(encoder="nope")
    except ValueError as exc:
        assert "Unknown sparse encoder" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
