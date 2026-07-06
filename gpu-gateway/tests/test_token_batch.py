"""Unit tests for gpu_gateway.token_batch — pure Python, no torch/ray/GPU needed.

Covers: split_token_aware's item-count cap, its token-budget cap (the padding-
amplification case: a batch of long chunks must NOT all ride in one sub-batch),
the single-oversized-item pass-through, estimate_tokens (heuristic + tokenizer),
and embed_token_budget's env-override parsing.

Run: uv run --no-project --with pytest --with pytest-asyncio pytest tests/ (PYTHONPATH=src)
"""

from __future__ import annotations

import pytest
from gpu_gateway.token_batch import (
    DEFAULT_EMBED_TOKEN_BUDGET,
    embed_token_budget,
    estimate_tokens,
    is_cuda_oom,
    split_token_aware,
)

# --------------------------------------------------------------------------
# estimate_tokens
# --------------------------------------------------------------------------


def test_estimate_tokens_chars_over_four_heuristic():
    assert estimate_tokens("a" * 40) == 10
    assert estimate_tokens("") == 1  # floor of 1, never zero
    assert estimate_tokens("abc") == 1  # 3 // 4 == 0, floored to 1


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        # A tokenizer's count need not match chars/4 — that's the point of preferring it.
        return list(range(len(text.split())))


def test_estimate_tokens_prefers_tokenizer_when_given():
    tok = _FakeTokenizer()
    assert estimate_tokens("one two three four five", tokenizer=tok) == 5
    # chars/4 heuristic gives a different count than the tokenizer for this string —
    # confirms the tokenizer path is actually used, not silently ignored.
    assert estimate_tokens("one two three four five") == len("one two three four five") // 4


class _BrokenTokenizer:
    def encode(self, text, add_special_tokens=True):
        raise RuntimeError("tokenizer exploded")


def test_estimate_tokens_falls_back_when_tokenizer_raises():
    assert estimate_tokens("a" * 40, tokenizer=_BrokenTokenizer()) == 10


# --------------------------------------------------------------------------
# split_token_aware — item-count cap (pre-existing behavior, must be unchanged)
# --------------------------------------------------------------------------


def test_split_respects_max_items_when_tokens_are_cheap():
    texts = [f"t{i}" for i in range(10)]
    batches = split_token_aware(texts, max_items=4, max_tokens=1_000_000)
    assert [len(b) for b in batches] == [4, 4, 2]
    assert sum(batches, []) == texts  # order + content preserved


def test_split_single_batch_when_under_both_caps():
    texts = ["hello", "world", "foo"]
    batches = split_token_aware(texts, max_items=32, max_tokens=DEFAULT_EMBED_TOKEN_BUDGET)
    assert batches == [texts]


def test_split_empty_input():
    assert split_token_aware([], max_items=32) == []


# --------------------------------------------------------------------------
# split_token_aware — token-budget cap (the padding-amplification regression case)
# --------------------------------------------------------------------------


def test_split_respects_token_budget_long_docling_chunks():
    """24 long chunks (~2048 est. tokens each, like the incident) under an item cap
    of 32 must NOT land in one sub-batch — the token budget must split it further."""
    long_chunk = (
        "word " * 2048
    )  # ~2048 tokens at chars/4 (5 chars/word ~ 1.25 tok/word... use exact calc)
    tok_per_chunk = estimate_tokens(long_chunk)
    texts = [long_chunk] * 24
    batches = split_token_aware(texts, max_items=32, max_tokens=8192)
    assert len(batches) > 1, "long chunks must be split across multiple sub-batches"
    for b in batches:
        total = sum(estimate_tokens(t) for t in b)
        assert total <= 8192 or len(b) == 1
    # sanity: with ~2048 tok/chunk and an 8192 budget, expect ~4 items per sub-batch
    assert all(len(b) <= (8192 // tok_per_chunk) + 1 for b in batches)


def test_split_token_budget_binds_even_under_item_cap():
    # 3 items of 3000 est. tokens each (chars/4): item cap of 32 never binds,
    # but the 8192 token budget must still force a split after 2 items.
    chunk = "x" * 12000  # 12000 // 4 == 3000 tokens
    texts = [chunk, chunk, chunk]
    batches = split_token_aware(texts, max_items=32, max_tokens=8192)
    assert len(batches) == 2
    assert [len(b) for b in batches] == [2, 1]


def test_split_single_oversized_item_gets_its_own_batch():
    """An item that alone exceeds the token budget still gets a one-item batch —
    splitting can't shrink a single input further; OOM recovery is the backstop."""
    huge = "x" * 100_000  # 25000 est. tokens, alone bigger than the 8192 budget
    texts = ["short", huge, "short2"]
    batches = split_token_aware(texts, max_items=32, max_tokens=8192)
    assert [huge] in batches
    huge_batch = next(b for b in batches if b == [huge])
    assert len(huge_batch) == 1


# --------------------------------------------------------------------------
# embed_token_budget — env override (config convention: env var, sensible default)
# --------------------------------------------------------------------------


def test_embed_token_budget_default(monkeypatch):
    monkeypatch.delenv("GPU_GATEWAY_EMBED_TOKEN_BUDGET", raising=False)
    assert embed_token_budget() == DEFAULT_EMBED_TOKEN_BUDGET == 8192


def test_embed_token_budget_env_override(monkeypatch):
    monkeypatch.setenv("GPU_GATEWAY_EMBED_TOKEN_BUDGET", "4096")
    assert embed_token_budget() == 4096


@pytest.mark.parametrize("bad", ["not-a-number", "0", "-5", ""])
def test_embed_token_budget_invalid_env_falls_back_to_default(monkeypatch, bad):
    if bad:
        monkeypatch.setenv("GPU_GATEWAY_EMBED_TOKEN_BUDGET", bad)
    else:
        monkeypatch.delenv("GPU_GATEWAY_EMBED_TOKEN_BUDGET", raising=False)
    assert embed_token_budget() == DEFAULT_EMBED_TOKEN_BUDGET


# --------------------------------------------------------------------------
# is_cuda_oom
# --------------------------------------------------------------------------


def test_is_cuda_oom_matches_torch_exception_type():
    torch = pytest.importorskip("torch")
    exc = torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 7.07 GiB")
    assert is_cuda_oom(exc)


def test_is_cuda_oom_matches_message_sniff_without_torch_type():
    exc = RuntimeError("CUDA out of memory. Tried to allocate 7.07 GiB")
    assert is_cuda_oom(exc)


def test_is_cuda_oom_false_for_unrelated_error():
    assert not is_cuda_oom(RuntimeError("connection refused"))
    assert not is_cuda_oom(ValueError("bad input"))
