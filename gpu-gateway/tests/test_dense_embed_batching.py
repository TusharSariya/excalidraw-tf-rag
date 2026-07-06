"""Unit tests for the dense-embed handlers' token-aware sub-batching + CUDA OOM
recovery (gpu_gateway.handlers.dense_embed._DenseEmbedBase).

Instantiates `_DenseEmbedBase` (and thin real subclasses) directly, bypassing the
`@serve.deployment` decorator and Ray entirely, with a FakeModel standing in for
`SentenceTransformer` — so these run on a GPU-less laptop with torch/CUDA mocked,
never touching a real device. Covers: halve-and-retry on OOM, single-item OOM
raising EmbedOOMError instead of crashing, and unchanged default output when the
token budget never binds (the pre-existing fixed-batch_size behavior).

Run: uv run --no-project --with pytest --with pytest-asyncio pytest tests/ (PYTHONPATH=src)
"""

from __future__ import annotations

import pytest
from gpu_gateway.handlers.dense_embed import (
    DenseEmbed06B,
    DenseEmbed4B,
    EmbedOOMError,
    _DenseEmbedBase,
)


class _Arr(list):
    """Stand-in for the numpy array SentenceTransformer.encode() returns."""

    def tolist(self):
        return list(self)


class _CudaOOM(RuntimeError):
    """Stand-in for torch.cuda.OutOfMemoryError — matched by token_batch.is_cuda_oom
    via the message-sniff fallback so this test needs no real torch/CUDA import."""

    def __init__(self, msg="CUDA out of memory. Tried to allocate 7.07 GiB"):
        super().__init__(msg)


class FakeModel:
    """Records every encode() call; `oom_if` decides per-call whether to raise."""

    def __init__(self, oom_if=None):
        self.calls: list[tuple[int, int]] = []  # (len(texts), batch_size)
        self._oom_if = oom_if or (lambda texts, batch_size: False)
        self.tokenizer = None

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32):
        self.calls.append((len(texts), batch_size))
        if self._oom_if(texts, batch_size):
            raise _CudaOOM()
        return _Arr([[float(len(t))] for t in texts])


class _FakeHandler(_DenseEmbedBase):
    """Directly instantiable stand-in — skips reconfigure()/Ray/serve.deployment."""

    _batch_size = 4

    def __init__(self, model: FakeModel) -> None:
        self._model = model
        self._model_path = "fake/model"
        self.empty_cache_calls = 0

    def _empty_cuda_cache(self) -> None:
        # Mock torch/CUDA entirely — no real torch.cuda.empty_cache() call.
        self.empty_cache_calls += 1


def _handler(oom_if=None, batch_size=4) -> tuple[_FakeHandler, FakeModel]:
    model = FakeModel(oom_if=oom_if)
    handler = _FakeHandler(model)
    handler._batch_size = batch_size
    return handler, model


# --------------------------------------------------------------------------
# Default behavior unchanged when the feature is at defaults (no OOM, texts short)
# --------------------------------------------------------------------------


async def test_call_returns_one_vector_per_text_in_order():
    handler, model = _handler()
    texts = [f"text-{i}" for i in range(10)]
    out = await handler(texts)
    assert len(out) == 10
    assert out == [[float(len(t))] for t in texts]


async def test_call_splits_by_item_count_cap_like_before():
    """10 texts with _batch_size=4 and a huge token budget must split into
    groups of <=4 items each (the pre-existing item-count batching), never OOM."""
    handler, model = _handler(batch_size=4)
    texts = [f"t{i}" for i in range(10)]
    out = await handler(texts)
    assert out == [[float(len(t))] for t in texts]
    assert [c[0] for c in model.calls] == [4, 4, 2]
    assert all(bs == n for n, bs in model.calls)  # batch_size == len(sub_batch), as before


async def test_call_single_subbatch_when_under_caps():
    handler, model = _handler(batch_size=32)
    texts = ["a", "b", "c"]
    out = await handler(texts)
    assert out == [[1.0], [1.0], [1.0]]
    assert model.calls == [(3, 3)]


# --------------------------------------------------------------------------
# Token-budget cap: long chunks force extra sub-batches even under the item cap
# --------------------------------------------------------------------------


async def test_call_token_budget_splits_long_chunks(monkeypatch):
    monkeypatch.setenv("GPU_GATEWAY_EMBED_TOKEN_BUDGET", "20")  # tiny, deterministic budget
    handler, model = _handler(batch_size=32)
    texts = ["x" * 40, "x" * 40, "x" * 40]  # 10 est. tokens each; budget 20 -> 2 per sub-batch
    out = await handler(texts)
    assert len(out) == 3
    assert [c[0] for c in model.calls] == [2, 1]


# --------------------------------------------------------------------------
# OOM recovery: halve-and-retry
# --------------------------------------------------------------------------


async def test_encode_sub_batch_halves_and_retries_on_oom():
    """OOM on any sub-batch bigger than 1 item; single items always succeed.
    A 4-item sub-batch must recursively halve down to four 1-item calls."""

    def oom_if(texts, batch_size):
        return len(texts) > 1

    handler, model = _handler(oom_if=oom_if)
    result = handler._encode_sub_batch(["a", "bb", "ccc", "dddd"], batch_size=4)
    assert result == [[1.0], [2.0], [3.0], [4.0]]
    # empty_cache is called once per OOM caught: 4-item fails, two 2-item fail -> 3 catches
    assert handler.empty_cache_calls == 3
    # every surviving call is single-item (batch_size halved down to 1)
    single_item_calls = [c for c in model.calls if c[0] == 1]
    assert len(single_item_calls) == 4


async def test_encode_sub_batch_halved_batch_size_matches_halved_items():
    calls_seen = []

    def oom_if(texts, batch_size):
        calls_seen.append((len(texts), batch_size))
        return len(texts) == 8

    handler, model = _handler(oom_if=oom_if, batch_size=8)
    result = handler._encode_sub_batch(["t"] * 8, batch_size=8)
    assert len(result) == 8
    # First call OOMs at (8, 8); recovery must halve BOTH item count and batch_size.
    assert calls_seen[0] == (8, 8)
    assert all(n <= 4 for n, _bs in calls_seen[1:])


# --------------------------------------------------------------------------
# Single-item OOM -> structured error, not a crash
# --------------------------------------------------------------------------


async def test_single_item_oom_raises_embed_oom_error_not_crash():
    def oom_if(texts, batch_size):
        return True  # always OOMs, even at a single item

    handler, model = _handler(oom_if=oom_if)
    with pytest.raises(EmbedOOMError):
        handler._encode_sub_batch(["poison"], batch_size=1)
    # cache was cleared before giving up — VRAM never left dirty
    assert handler.empty_cache_calls == 1


async def test_call_isolates_single_poison_item_via_recursive_halving():
    """One OOM'ing item inside a larger batch surfaces EmbedOOMError for that
    sub-batch without ever raising an unrelated exception type."""

    def oom_if(texts, batch_size):
        return "poison" in texts

    handler, model = _handler(oom_if=oom_if, batch_size=4)
    with pytest.raises(EmbedOOMError):
        await handler(["fine", "poison", "also-fine", "still-fine"])


async def test_non_oom_exception_propagates_unchanged():
    def oom_if(texts, batch_size):
        raise ValueError("not an OOM at all")

    handler, model = _handler(oom_if=oom_if)
    with pytest.raises(ValueError):
        handler._encode_sub_batch(["a"], batch_size=1)
    assert handler.empty_cache_calls == 0  # never treated as OOM, no cache clear


# --------------------------------------------------------------------------
# Real subclasses keep their per-model batch_size defaults (regression guard)
# --------------------------------------------------------------------------


def test_subclass_batch_size_defaults_unchanged():
    # DenseEmbed4B / DenseEmbed06B are ray.serve.deployment-wrapped; the underlying
    # class is reachable via .func_or_class for a decorator-bypassing assertion.
    assert DenseEmbed4B.func_or_class._batch_size == 8
    assert DenseEmbed06B.func_or_class._batch_size == 4
