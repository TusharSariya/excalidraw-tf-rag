"""Token-aware embed sub-batching + CUDA OOM recovery.

`SentenceTransformer.encode()` pads every sequence in a mini-batch to the length of
the longest sequence in that mini-batch, so effective attention memory scales with
``batch_size x longest_seq**2`` — not with item count alone. A batch of 24 long
docling chunks (each near the model's ``max_seq_length``) produced a 7.07 GB CUDA
allocation on an 8 GB RTX 3060 Ti overnight even though the handlers already cap
mini-batches by item count (32 / 8 / 4 — see ``handlers/dense_embed.py``).

This module is deliberately torch-free at import time (mirrors ``scheduler.py``'s
"pure orchestration, GPU-touching bits injected/lazily-imported" convention) so it
stays unit-testable on a GPU-less laptop:

- ``split_token_aware`` groups incoming texts into sub-batches capped by BOTH an
  item count AND an estimated total token budget, so a handful of long chunks
  never rides along with the full item-count allowance.
- ``EmbedOOMError`` is raised by the handler's OOM-recovery retry loop (see
  ``dense_embed.py::_encode_sub_batch``) when a single item still OOMs after
  halving — the caller (``main.py``) turns this into a structured per-request
  HTTP error instead of letting the exception crash the Ray Serve replica.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger("gpu_gateway.token_batch")

# Default token budget per embed sub-batch, env-overridable via
# GPU_GATEWAY_EMBED_TOKEN_BUDGET. Derived for the 8 GB RTX 3060 Ti target
# (_CARD_MB=7200 in main.py): the incident that motivated this module summed to
# roughly ~49k estimated tokens (24 chunks near a ~2048-token max_seq_length).
# Capping a sub-batch's summed estimated tokens at 8192 — the same figure already
# used for the scheduler's coalescing bound (DRAIN_TOKEN_BUDGET in scheduler.py) —
# keeps a sub-batch to a handful of long chunks (or dozens of short ones) rather
# than dozens of long ones, comfortably under budget even for the 4B handler's
# tighter per-layer activation headroom (~1.5 GB free after loading int8 weights).
DEFAULT_EMBED_TOKEN_BUDGET = 8192

_ENV_TOKEN_BUDGET = "GPU_GATEWAY_EMBED_TOKEN_BUDGET"


class EmbedOOMError(RuntimeError):
    """A single item could not be embedded even alone (after a cache-clear retry).

    Handlers raise this instead of letting ``torch.cuda.OutOfMemoryError`` (or an
    equivalent CUDA allocation failure) propagate and crash the Ray Serve replica;
    ``GatewayHTTP`` catches it and returns a structured 4xx/5xx JSON error for that
    request while the process — and the rest of a coalesced batch — keeps running.
    """


def embed_token_budget() -> int:
    """Resolve the per-sub-batch token budget: GPU_GATEWAY_EMBED_TOKEN_BUDGET env
    override if set and valid, else DEFAULT_EMBED_TOKEN_BUDGET."""
    raw = os.environ.get(_ENV_TOKEN_BUDGET)
    if not raw:
        return DEFAULT_EMBED_TOKEN_BUDGET
    try:
        val = int(raw)
    except ValueError:
        log.warning(
            "invalid %s=%r; using default %d", _ENV_TOKEN_BUDGET, raw, DEFAULT_EMBED_TOKEN_BUDGET
        )
        return DEFAULT_EMBED_TOKEN_BUDGET
    if val <= 0:
        log.warning(
            "%s=%d must be positive; using default %d",
            _ENV_TOKEN_BUDGET,
            val,
            DEFAULT_EMBED_TOKEN_BUDGET,
        )
        return DEFAULT_EMBED_TOKEN_BUDGET
    return val


def estimate_tokens(text: str, tokenizer=None) -> int:
    """Estimate the token count of one string.

    Uses the model's tokenizer when one is loaded (accurate); falls back to a
    chars/4 heuristic otherwise — coarse but cheap, and the same proxy already
    used for the scheduler's coalescing token estimate (see ``Job.__post_init__``
    in scheduler.py), so the two layers agree on what "a token" costs.
    """
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=True))
        except Exception as exc:  # noqa: BLE001 — tokenizer quirks shouldn't break batching
            log.debug("tokenizer.encode failed (%s); falling back to chars/4", exc)
    return max(1, len(text) // 4)


def split_token_aware(
    texts: list[str],
    *,
    max_items: int,
    max_tokens: int | None = None,
    tokenizer=None,
) -> list[list[str]]:
    """Split `texts` into ordered sub-batches capped by BOTH `max_items` and an
    estimated total token budget (`max_tokens`, default `embed_token_budget()`).

    A single item that alone exceeds `max_tokens` still gets its own one-item
    sub-batch — splitting can't shrink one input any further; OOM recovery on
    that sub-batch (see `dense_embed.py::_encode_sub_batch`) is the safety net.
    """
    if max_tokens is None:
        max_tokens = embed_token_budget()
    if not texts:
        return []
    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for text in texts:
        tok = estimate_tokens(text, tokenizer)
        exceeds_tokens = bool(current) and (current_tokens + tok > max_tokens)
        exceeds_items = len(current) >= max(1, max_items)
        if current and (exceeds_tokens or exceeds_items):
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(text)
        current_tokens += tok
    if current:
        batches.append(current)
    return batches


def is_cuda_oom(exc: BaseException) -> bool:
    """True if `exc` looks like a CUDA out-of-memory failure.

    Prefers an exact `torch.cuda.OutOfMemoryError` isinstance check (torch is
    imported lazily here, matching the rest of the package's convention of
    keeping GPU-touching imports out of module top-level); falls back to a
    message sniff so a wrapped/re-raised OOM (e.g. from bitsandbytes) is still
    recognized.
    """
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:
        pass
    msg = str(exc).lower()
    return "out of memory" in msg and "cuda" in msg
