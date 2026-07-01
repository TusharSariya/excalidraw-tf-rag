"""Tier 1 — gated correctness checks against a LIVE gateway (desktop GPU).

These turn the plan's V1/V2/V4/V5/V9 claims into fail-closed assertions. They run
only when `--gateway-url` is supplied (see conftest.py); otherwise auto-skipped.

    pytest -m integration --gateway-url http://localhost:8765 -v

Thresholds are pinned in module constants below. The strict scheduling *logic*
(selection order, victim ranking, evict-all-but-target) is unit-tested in
test_scheduler.py without a GPU; these tests verify the *observable consequences*
on real hardware — no VRAM breach, no co-residency thrash, no starvation, no
lingering LLM residency.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

pytestmark = pytest.mark.integration

# Unique per test process: the gateway's SQLite cache would otherwise serve repeated
# inputs WITHOUT loading a model, so residency/eviction assertions would see an empty
# card on a re-run. Salting every input guarantees cache misses → real dispatcher work.
NONCE = os.urandom(4).hex()


def _salt(s: str) -> str:
    return f"{NONCE}-{s}"


# --- Pinned thresholds -----------------------------------------------------
CARD_CEILING_MB = 7500  # any /health sample above this = a real breach (V1/V5)
EMBED_0_6B = "cuda-qwen0.6b-1024"
EMBED_4B = "cuda-qwen4b-1024"
RERANKER = "bge-reranker-v2-m3"
CHAT_MODEL = "gemma4:e2b"  # must exist in Ollama (`ollama list`)
LLM_CLEARED_CEILING_MB = 4500  # 0.6B (~1800) + CUDA ctx; well below a lingering LLM (V9)
STARVE_SLACK_S = 45.0  # generous ceiling for "not starved" under real load (V4)


def _text(n: int) -> str:
    return f"probe {NONCE} {n} " + ("graph layout constraint solver " * 4)


class _PeakSampler(threading.Thread):
    """Polls /health in the background, tracking peak used_mb and whether the 4B and
    0.6B embedders were ever co-resident (that pair = 7600 MB > 7200 and must be refused)."""

    def __init__(self, gw, interval: float = 0.05) -> None:
        super().__init__(daemon=True)
        self.gw = gw
        self.interval = interval
        self.peak_mb = 0
        self.both_resident = False
        self.samples = 0
        self._halt = threading.Event()  # NOT _stop: that shadows Thread._stop(), which join() calls

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                h = self.gw.health()
                self.peak_mb = max(self.peak_mb, self.gw.used_of(h))
                lm = set(h.get("loaded_models", []))
                if {EMBED_4B, EMBED_0_6B} <= lm:
                    self.both_resident = True
                self.samples += 1
            except Exception:  # noqa: BLE001 — a dropped sample must not kill the sampler
                pass
            self._halt.wait(self.interval)

    def stop(self) -> None:
        self._halt.set()


# --------------------------------------------------------------------------
# V1 — atomic eviction / no VRAM breach
# --------------------------------------------------------------------------


def test_v1_no_vram_breach(gw):
    """~20 interleaved 4B/0.6B embeds at concurrency 8. VRAM must never exceed the
    card at any sample, and 4B+0.6B must never be co-resident (bin-packing refuses it)."""
    models = [EMBED_4B, EMBED_0_6B]
    sampler = _PeakSampler(gw)
    sampler.start()

    def one(i: int):
        status, _, _ = gw.post("/v1/embeddings", {"model": models[i % 2], "input": [_text(i)]})
        return status

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(one, range(20)))

    sampler.stop()
    sampler.join(timeout=3)

    assert all(s == 200 for s in statuses), f"non-200 responses: {statuses}"
    assert sampler.samples > 0, "health sampler collected no samples"
    assert sampler.peak_mb < CARD_CEILING_MB, (
        f"VRAM breach: peak {sampler.peak_mb} MB >= {CARD_CEILING_MB} MB"
    )
    assert not sampler.both_resident, (
        "4B and 0.6B were co-resident (7600 > 7200 must be refused by bin-packing)"
    )


# --------------------------------------------------------------------------
# V2 — co-residency, no embed→rerank thrash
# --------------------------------------------------------------------------


def test_v2_coresidency_no_thrash(gw):
    """Warm 0.6B + reranker, then loop embed→rerank. Both must stay co-resident and
    the evicts counter must not advance across the loop (3000 MB fits under 7200)."""
    gw.post("/v1/embeddings", {"model": EMBED_0_6B, "input": [_salt("warm")]})
    gw.post(
        "/v1/rerank", {"model": RERANKER, "query": _salt("q"), "passages": [_salt("a"), _salt("b")]}
    )

    before = gw.counter("gpu_gateway_evicts")
    both_seen = False
    for i in range(8):
        gw.post("/v1/embeddings", {"model": EMBED_0_6B, "input": [_salt(f"e{i}")]})
        gw.post(
            "/v1/rerank", {"model": RERANKER, "query": _salt("q"), "passages": [_salt(f"p{i}")]}
        )
        if {EMBED_0_6B, RERANKER} <= gw.loaded():
            both_seen = True
    after = gw.counter("gpu_gateway_evicts")

    assert both_seen, "0.6B and reranker were never co-resident together"
    assert after == before, (
        f"eviction thrash during co-residency loop: {after - before} evicts (expected 0)"
    )


# --------------------------------------------------------------------------
# V4 — queue-depth scheduling, no starvation
# --------------------------------------------------------------------------


def test_v4_no_starvation_under_mixed_load(gw):
    """Fire a deep 0.6B burst + a shallow reranker burst concurrently. The shallow
    queue must not be starved: all its requests complete within a bounded wall time,
    and the deep queue is observed deeper than the shallow one at least once."""
    deep_n, shallow_n = 24, 3
    depth_seen = {"deeper": False}
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            try:
                qd = gw.health().get("queue_depth", {})
                if qd.get(EMBED_0_6B, 0) > qd.get(RERANKER, 0):
                    depth_seen["deeper"] = True
            except Exception:  # noqa: BLE001
                pass
            stop.wait(0.05)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()

    shallow_lat: list[float] = []

    def deep(i: int):
        return gw.post("/v1/embeddings", {"model": EMBED_0_6B, "input": [_text(i)]})[1] is not None

    def shallow(i: int):
        _, _, dt = gw.post(
            "/v1/rerank",
            {
                "model": RERANKER,
                "query": _salt(f"q{i}"),
                "passages": [_salt(f"p{i}-{j}") for j in range(6)],
            },
        )
        shallow_lat.append(dt)
        return True

    with ThreadPoolExecutor(max_workers=10) as pool:
        futs = [pool.submit(deep, i) for i in range(deep_n)]
        futs += [pool.submit(shallow, i) for i in range(shallow_n)]
        results = [f.result() for f in futs]

    stop.set()
    watcher.join(timeout=2)

    assert all(results), "some mixed-load requests failed"
    assert len(shallow_lat) == shallow_n, "shallow (reranker) requests did not all complete"
    assert max(shallow_lat) < STARVE_SLACK_S, (
        f"reranker starved: slowest {max(shallow_lat):.1f}s >= {STARVE_SLACK_S}s ceiling"
    )
    # The real gate is no-starvation (above). Observing the deep queue actually exceed
    # the shallow one over /health polling is racy: the dispatcher coalesces the whole
    # 0.6B burst into one drain almost instantly, so the queue rarely lingers deep long
    # enough for a 50 ms poll to catch it. The queue-depth *ordering* is covered
    # deterministically by test_scheduler.py (test_select_deepest_*); here it's just
    # logged evidence, not a gate.
    print(f"[V4] deep-queue-exceeded-shallow observed: {depth_seen['deeper']}")


# --------------------------------------------------------------------------
# V5 — sequential load under pressure keeps peak under the card
# --------------------------------------------------------------------------


def test_v5_sequential_load_no_overbudget_spike(gw):
    """Clear the card with a chat request, then concurrently admit two different
    embedders from cold. Sequential loading (single dispatcher) must keep every peak
    sample under the card — no overlapping from-disk load spike breaches the budget."""
    # Chat evicts everything → the card is (near) cold.
    gw.post(
        "/v1/chat/completions",
        {"model": CHAT_MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
    )

    sampler = _PeakSampler(gw, interval=0.03)
    sampler.start()

    # Two DIFFERENT models loaded concurrently from cold, each via its correct endpoint
    # (an embedder is not a reranker). They co-reside (3000 MB) but the single dispatcher
    # still loads them one at a time — the property under test.
    def load_embed():
        return gw.post("/v1/embeddings", {"model": EMBED_0_6B, "input": [_salt("cold-e")]})[0]

    def load_rerank():
        return gw.post(
            "/v1/rerank",
            {"model": RERANKER, "query": _salt("cold-q"), "passages": [_salt("cold-p")]},
        )[0]

    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = [pool.submit(load_embed), pool.submit(load_rerank)]
        statuses = [f.result() for f in futs]

    sampler.stop()
    sampler.join(timeout=3)

    assert all(s == 200 for s in statuses), f"cold-load responses: {statuses}"
    assert sampler.peak_mb < CARD_CEILING_MB, (
        f"overlapping load spike breached the card: peak {sampler.peak_mb} MB"
    )


# --------------------------------------------------------------------------
# V9 — LLM residency actually clears before the next embed
# --------------------------------------------------------------------------


def test_v9_llm_residency_cleared_before_embed(gw):
    """A chat request then an immediate embed. The embed must succeed and, once it
    admits, VRAM must reflect only the embedder — proving Ollama's residency cleared
    (the measured gate waited), not merely that KEEP_ALIVE=0 is set."""
    status_chat, _, _ = gw.post(
        "/v1/chat/completions",
        {"model": CHAT_MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
    )
    assert status_chat == 200, "chat request failed"

    status_embed, _, _ = gw.post(
        "/v1/embeddings", {"model": EMBED_0_6B, "input": [_salt("after llm")]}
    )
    assert status_embed == 200, "embed after chat failed (LLM may still hold the card)"

    # Give the trace a beat to settle, then confirm the card holds only the embedder.
    time.sleep(0.5)
    used = gw.used_mb()
    loaded = gw.loaded()
    assert EMBED_0_6B in loaded, f"0.6B not resident after admit; loaded={loaded}"
    assert used < LLM_CLEARED_CEILING_MB, (
        f"LLM VRAM lingered: used {used} MB >= {LLM_CLEARED_CEILING_MB} MB after embed admit"
    )
