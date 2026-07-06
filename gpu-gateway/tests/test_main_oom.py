"""Unit tests for gpu_gateway.main._is_embed_oom — the HTTP-layer detector that
turns an EmbedOOMError (raised by a dense-embed handler after halve-and-retry
exhausts itself on a single item) into a structured per-request error instead of
a bare 500. Pure function, no Ray/GPU needed; module import itself needs no live
Ray runtime (only ray.get_actor is called lazily inside GatewayHTTP.__init__).

Run: uv run --no-project --with pytest --with pytest-asyncio pytest tests/ (PYTHONPATH=src)
"""

from __future__ import annotations

from gpu_gateway.handlers.dense_embed import EmbedOOMError
from gpu_gateway.main import _is_embed_oom


def test_is_embed_oom_direct_instance():
    assert _is_embed_oom(EmbedOOMError("boom"))


def test_is_embed_oom_false_for_unrelated_exception():
    assert not _is_embed_oom(RuntimeError("connection refused"))
    assert not _is_embed_oom(ValueError("bad input"))


def test_is_embed_oom_wrapped_via_cause():
    # Mirrors how the dense-embed handler chains EmbedOOMError from the original
    # CUDA exception (`raise EmbedOOMError(...) from exc`) and something upstream
    # (e.g. a Ray actor boundary re-raise) might further wrap it.
    original = EmbedOOMError("single item OOM'd")
    try:
        try:
            raise original
        except EmbedOOMError as exc:
            raise RuntimeError("actor call failed") from exc
    except RuntimeError as wrapped:
        assert _is_embed_oom(wrapped)


def test_is_embed_oom_wrapped_via_context():
    original = EmbedOOMError("single item OOM'd")
    try:
        try:
            raise original
        except EmbedOOMError:
            raise RuntimeError("actor call failed") from None  # implicit __context__ still set
    except RuntimeError as wrapped:
        assert _is_embed_oom(wrapped)


def test_is_embed_oom_class_name_fallback_across_reconstruction():
    """Simulate an exception that lost its original type across a serialization
    boundary (e.g. a Ray version that doesn't preserve isinstance) but kept the
    class name — the fallback must still recognize it. Uses a distinct dynamically
    named class (never mutates a shared builtin's __name__)."""
    reconstructed_cls = type("EmbedOOMError", (RuntimeError,), {})
    carrier = reconstructed_cls("single item OOM'd")
    fake = RuntimeError("wrapped")
    fake.__cause__ = carrier
    assert _is_embed_oom(fake)


def test_is_embed_oom_no_infinite_loop_on_self_referential_chain():
    exc = RuntimeError("a")
    exc.__cause__ = exc  # pathological but must not hang
    assert not _is_embed_oom(exc)
