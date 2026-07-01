"""Unit suite for the VRAM Scheduler decision logic (no GPU, no ray/torch).

Covers: model selection (affinity + deepest-queue + starvation), _admit bin-packing
(co-reside vs static-evict vs measured-gate-evict-more vs raise), _evict_for victim
ranking (idle → LRU → queue-depth), coalesced embed serve + per-job fallback, the
dispatcher error boundary + reconcile, and the _chat → evict-all regression.

Run: uv run --no-project --with pytest --with pytest-asyncio pytest tests/ (PYTHONPATH=src)
"""

from __future__ import annotations

import asyncio
import collections

import pytest
from gpu_gateway.scheduler import LLM_SENTINEL, Job, Scheduler

VRAM = {
    "cuda-qwen4b-1024": 5800,
    "cuda-qwen0.6b-1024": 1800,
    "bge-reranker-v2-m3": 1200,
}
CARD = 7200
DEVICE = "cuda:0"


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def make_sched(*, free=100_000, unload_parks=False, **kw) -> tuple[Scheduler, dict]:
    """Build a Scheduler with recording fakes. `free` may be an int or a 0-arg callable
    (to model measured VRAM that changes as models evict)."""
    rec = {"embed": [], "rerank": [], "chat": [], "unload": []}
    embed_raise = kw.pop("embed_raise", None)  # None | "always" | callable(texts)->bool

    async def embed_fn(canonical, texts):
        rec["embed"].append((canonical, list(texts)))
        if embed_raise == "always":
            raise RuntimeError("boom")
        if callable(embed_raise) and embed_raise(texts):
            raise RuntimeError("bad-batch")
        return [[float(len(t)), 0.0] for t in texts]

    async def rerank_fn(query, passages):
        rec["rerank"].append((query, list(passages)))
        return [float(len(p)) for p in passages]

    async def chat_fn(payload):
        rec["chat"].append(payload)
        return {"ok": True, "echo": payload.get("model")}

    async def unload_fn(canonical):
        rec["unload"].append(canonical)
        return unload_parks

    measured = free if callable(free) else (lambda: free)
    clock = kw.pop("clock", FakeClock())
    sched = Scheduler(
        vram_mb=VRAM,
        card_mb=CARD,
        embed_fn=embed_fn,
        rerank_fn=rerank_fn,
        chat_fn=chat_fn,
        unload_fn=unload_fn,
        measured_free_mb=measured,
        clock=clock,
        evict_poll_s=0.0,
        **kw,
    )
    return sched, rec


def enqueue(sched: Scheduler, model: str, n: int, kind: str = "embed", *, at: float = 0.0):
    """Populate a queue directly (bypassing the loop) for pure-logic tests.
    Futures are left None — selection/drain/victim logic never touches them."""
    q = sched._queues.setdefault(model, collections.deque())
    for i in range(n):
        q.append(Job(kind=kind, canonical=model, items=[f"t{i}"], enqueued_at=at))


# --------------------------------------------------------------------------
# _select_model
# --------------------------------------------------------------------------


def test_select_resident_with_work_beats_deeper_nonresident():
    sched, _ = make_sched()
    sched._resident[DEVICE] = {"cuda-qwen0.6b-1024"}
    enqueue(sched, "cuda-qwen0.6b-1024", 1)
    enqueue(sched, "cuda-qwen4b-1024", 3)  # deeper but NOT resident
    assert sched._select_model() == "cuda-qwen0.6b-1024"


def test_select_deepest_when_no_resident_work():
    sched, _ = make_sched()
    enqueue(sched, "cuda-qwen0.6b-1024", 2)
    enqueue(sched, "cuda-qwen4b-1024", 5)
    assert sched._select_model() == "cuda-qwen4b-1024"


def test_select_deepest_among_resident_with_work():
    sched, _ = make_sched()
    sched._resident[DEVICE] = {"cuda-qwen0.6b-1024", "bge-reranker-v2-m3"}
    enqueue(sched, "cuda-qwen0.6b-1024", 2)
    enqueue(sched, "bge-reranker-v2-m3", 5)
    assert sched._select_model() == "bge-reranker-v2-m3"


def test_select_none_when_empty():
    sched, _ = make_sched()
    assert sched._select_model() is None


def test_starvation_preempts_long_served_model():
    clock = FakeClock()
    sched, _ = make_sched(clock=clock)
    sched._resident[DEVICE] = {"cuda-qwen0.6b-1024"}
    # 0.6B has been served a long streak; a 4B job has waited past STARVE_MS.
    sched._last_served = "cuda-qwen0.6b-1024"
    sched._served_streak = sched._drain_max
    enqueue(sched, "cuda-qwen0.6b-1024", 1, at=clock.t)
    enqueue(sched, "cuda-qwen4b-1024", 1, at=clock.t)
    clock.advance(sched._starve_ms / 1000.0 + 1.0)  # age the 4B waiter past the floor
    assert sched._select_model() == "cuda-qwen4b-1024"


def test_no_starvation_before_streak():
    clock = FakeClock()
    sched, _ = make_sched(clock=clock)
    sched._resident[DEVICE] = {"cuda-qwen0.6b-1024"}
    sched._last_served = "cuda-qwen0.6b-1024"
    sched._served_streak = sched._drain_max - 1  # streak not yet reached
    enqueue(sched, "cuda-qwen0.6b-1024", 1, at=clock.t)
    enqueue(sched, "cuda-qwen4b-1024", 1, at=clock.t)
    clock.advance(sched._starve_ms / 1000.0 + 1.0)
    assert sched._select_model() == "cuda-qwen0.6b-1024"  # resident-with-work, no preempt


# --------------------------------------------------------------------------
# _drain
# --------------------------------------------------------------------------


def test_drain_coalesces_embed_up_to_max():
    sched, _ = make_sched(drain_max=4)
    enqueue(sched, "cuda-qwen0.6b-1024", 10)
    batch = sched._drain("cuda-qwen0.6b-1024")
    assert len(batch) == 4
    assert len(sched._queues["cuda-qwen0.6b-1024"]) == 6


def test_drain_token_budget_stops_early():
    sched, _ = make_sched(drain_max=100, drain_token_budget=10)
    q = sched._queues.setdefault("cuda-qwen0.6b-1024", collections.deque())
    for _ in range(5):
        q.append(
            Job(kind="embed", canonical="cuda-qwen0.6b-1024", items=["x" * 40])
        )  # ~10 tokens each
    batch = sched._drain("cuda-qwen0.6b-1024")
    # first job always taken; second would exceed the 10-token budget → stop at 1
    assert len(batch) == 1


def test_drain_rerank_one_at_a_time():
    sched, _ = make_sched()
    enqueue(sched, "bge-reranker-v2-m3", 3, kind="rerank")
    batch = sched._drain("bge-reranker-v2-m3")
    assert len(batch) == 1


# --------------------------------------------------------------------------
# _pick_victim
# --------------------------------------------------------------------------


def test_victim_idle_before_busy():
    sched, _ = make_sched()
    sched._resident[DEVICE] = {"cuda-qwen0.6b-1024", "bge-reranker-v2-m3"}
    sched._loaded_at = {"cuda-qwen0.6b-1024": 1.0, "bge-reranker-v2-m3": 2.0}
    enqueue(sched, "cuda-qwen0.6b-1024", 2)  # busy
    # reranker idle → evict reranker even though it's more recently used
    assert sched._pick_victim("cuda-qwen4b-1024") == "bge-reranker-v2-m3"


def test_victim_lru_among_idle():
    sched, _ = make_sched()
    sched._resident[DEVICE] = {"cuda-qwen0.6b-1024", "bge-reranker-v2-m3"}
    sched._loaded_at = {"cuda-qwen0.6b-1024": 1.0, "bge-reranker-v2-m3": 2.0}
    assert sched._pick_victim("cuda-qwen4b-1024") == "cuda-qwen0.6b-1024"  # oldest


def test_victim_shallower_queue_tiebreak():
    sched, _ = make_sched()
    sched._resident[DEVICE] = {"cuda-qwen0.6b-1024", "bge-reranker-v2-m3"}
    sched._loaded_at = {"cuda-qwen0.6b-1024": 5.0, "bge-reranker-v2-m3": 5.0}
    enqueue(sched, "cuda-qwen0.6b-1024", 1)
    enqueue(sched, "bge-reranker-v2-m3", 4)
    assert sched._pick_victim("cuda-qwen4b-1024") == "cuda-qwen0.6b-1024"  # shallower


# --------------------------------------------------------------------------
# _admit
# --------------------------------------------------------------------------


async def test_admit_coresidence_no_evict():
    sched, rec = make_sched()
    sched._resident[DEVICE] = {"cuda-qwen0.6b-1024"}  # 1800
    await sched._admit("bge-reranker-v2-m3")  # +1200 = 3000 ≤ 7200
    assert sched._resident[DEVICE] == {"cuda-qwen0.6b-1024", "bge-reranker-v2-m3"}
    assert rec["unload"] == []


async def test_admit_evicts_when_over_static_budget():
    sched, rec = make_sched()
    sched._resident[DEVICE] = {"cuda-qwen4b-1024"}  # 5800
    sched._loaded_at = {"cuda-qwen4b-1024": 1.0}
    await sched._admit("cuda-qwen0.6b-1024")  # 5800+1800=7600 > 7200 → evict 4B
    assert rec["unload"] == ["cuda-qwen4b-1024"]
    assert sched._resident[DEVICE] == {"cuda-qwen0.6b-1024"}


async def test_admit_measured_gate_evicts_more():
    # Static budget says 4B fits next to the reranker (1200+5800=7000 ≤ 7200), but
    # measured free is low while the reranker is resident → gate must evict it too.
    state = {"reranker_resident": True}

    def measured():
        return 1000 if state["reranker_resident"] else 6800

    sched, rec = make_sched(free=measured)
    sched._resident[DEVICE] = {"bge-reranker-v2-m3"}
    sched._loaded_at = {"bge-reranker-v2-m3": 1.0}

    async def unload_fn(canonical):
        rec["unload"].append(canonical)
        if canonical == "bge-reranker-v2-m3":
            state["reranker_resident"] = False
        return False

    sched._unload_fn = unload_fn
    await sched._admit("cuda-qwen4b-1024")
    assert rec["unload"] == ["bge-reranker-v2-m3"]  # measured gate forced the evict
    assert sched._resident[DEVICE] == {"cuda-qwen4b-1024"}


async def test_admit_raises_when_nothing_to_evict():
    sched, _ = make_sched(free=100)  # perpetually too low, nothing resident to evict
    with pytest.raises(RuntimeError):
        await sched._admit("cuda-qwen4b-1024")


async def test_admit_already_resident_is_noop():
    sched, rec = make_sched()
    sched._resident[DEVICE] = {"cuda-qwen4b-1024"}
    await sched._admit("cuda-qwen4b-1024")
    assert rec["unload"] == []


# --------------------------------------------------------------------------
# Naive mode (GATEWAY_SCHED_MODE=naive) — the A/B baseline: single-resident
# load-evict, no coalescing. These lock the control curve so the wedge proof's
# comparison is trustworthy.
# --------------------------------------------------------------------------


def test_naive_drain_serves_one_even_for_embed():
    # Embed is coalescable, but naive must serve exactly one request per load.
    sched, _ = make_sched(mode="naive", drain_max=64)
    enqueue(sched, "cuda-qwen0.6b-1024", 10)
    batch = sched._drain("cuda-qwen0.6b-1024")
    assert len(batch) == 1
    assert len(sched._queues["cuda-qwen0.6b-1024"]) == 9


async def test_naive_refuses_coresidence_that_affinity_allows():
    # Same setup as test_admit_coresidence_no_evict (0.6B+reranker=3000 ≤ 7200),
    # but naive evicts the incumbent instead of co-residing.
    sched, rec = make_sched(mode="naive")
    sched._resident[DEVICE] = {"cuda-qwen0.6b-1024"}
    sched._loaded_at = {"cuda-qwen0.6b-1024": 1.0}
    await sched._admit("bge-reranker-v2-m3")
    assert rec["unload"] == ["cuda-qwen0.6b-1024"]
    assert sched._resident[DEVICE] == {"bge-reranker-v2-m3"}


async def test_naive_admit_evicts_all_other_residents():
    sched, rec = make_sched(mode="naive")
    sched._resident[DEVICE] = {"cuda-qwen0.6b-1024", "bge-reranker-v2-m3"}
    sched._loaded_at = {"cuda-qwen0.6b-1024": 1.0, "bge-reranker-v2-m3": 2.0}
    await sched._admit("cuda-qwen4b-1024")
    assert set(rec["unload"]) == {"cuda-qwen0.6b-1024", "bge-reranker-v2-m3"}
    assert sched._resident[DEVICE] == {"cuda-qwen4b-1024"}


def test_affinity_drain_still_coalesces():
    # Guard the default path: affinity still coalesces embed up to drain_max.
    sched, _ = make_sched(drain_max=64)  # mode defaults to "affinity"
    enqueue(sched, "cuda-qwen0.6b-1024", 10)
    assert len(sched._drain("cuda-qwen0.6b-1024")) == 10


# --------------------------------------------------------------------------
# _serve — coalescing + fallback
# --------------------------------------------------------------------------


async def test_serve_coalesces_embed_into_one_call():
    sched, rec = make_sched()
    jobs = []
    for i in range(3):
        j = Job(kind="embed", canonical="cuda-qwen0.6b-1024", items=[f"a{i}", f"b{i}"])
        j.future = asyncio.get_running_loop().create_future()
        jobs.append(j)
    await sched._serve("cuda-qwen0.6b-1024", jobs)
    assert len(rec["embed"]) == 1  # ONE coalesced call
    assert len(rec["embed"][0][1]) == 6  # all 3×2 texts
    for j in jobs:
        assert len(j.future.result()) == 2  # each job gets its 2 vectors back


async def test_serve_per_job_fallback_on_batch_error():
    # Coalesced call fails only when the combined batch contains the poison text;
    # per-job retry should still resolve the clean jobs and fail only the bad one.
    def poison(texts):
        return "POISON" in texts and len(texts) > 1  # fail the coalesced call

    sched, rec = make_sched(embed_raise=poison)
    good = Job(kind="embed", canonical="cuda-qwen0.6b-1024", items=["ok"])
    bad = Job(kind="embed", canonical="cuda-qwen0.6b-1024", items=["POISON"])
    for j in (good, bad):
        j.future = asyncio.get_running_loop().create_future()
    await sched._serve("cuda-qwen0.6b-1024", [good, bad])
    assert good.future.result() is not None  # clean job resolved via fallback
    # bad job: single-item call has len==1 so poison() is False → it also resolves
    assert bad.future.done()
    assert sched._metrics["coalesce_fallbacks"] == 1


# --------------------------------------------------------------------------
# End-to-end dispatcher (real event loop)
# --------------------------------------------------------------------------


async def test_end_to_end_embed_roundtrip():
    sched, rec = make_sched()
    res = await sched.submit(Job(kind="embed", canonical="cuda-qwen0.6b-1024", items=["hello"]))
    assert res == [[5.0, 0.0]]
    assert sched._resident[DEVICE] == {"cuda-qwen0.6b-1024"}


async def test_chat_evicts_all_regression():
    """A chat request must evict every resident model, serve the LLM alone, then drop
    the sentinel so the next embed admit re-checks VRAM."""
    sched, rec = make_sched()
    # Pretend two models are already resident.
    sched._resident[DEVICE] = {"cuda-qwen0.6b-1024", "bge-reranker-v2-m3"}
    sched._loaded_at = {"cuda-qwen0.6b-1024": 1.0, "bge-reranker-v2-m3": 2.0}

    res = await sched.submit(
        Job(kind="chat", canonical=LLM_SENTINEL, payload={"model": "gemma4:e4b"})
    )
    assert res["ok"] is True
    assert set(rec["unload"]) == {"cuda-qwen0.6b-1024", "bge-reranker-v2-m3"}
    assert sched._resident[DEVICE] == set()  # sentinel dropped post-serve
    assert len(rec["chat"]) == 1


async def test_error_boundary_fails_batch_and_survives():
    """A load/serve failure fails that request's future but the dispatcher keeps
    running and serves the next request."""
    sched, rec = make_sched(embed_raise="always")
    with pytest.raises(RuntimeError):
        await sched.submit(Job(kind="embed", canonical="cuda-qwen0.6b-1024", items=["x"]))
    assert sched.alive()  # loop survived the failure
    # Now make embeds succeed and confirm the loop still serves.
    ok_calls = {"n": 0}

    async def embed_ok(canonical, texts):
        ok_calls["n"] += 1
        return [[1.0] for _ in texts]

    sched._embed_fn = embed_ok
    res = await sched.submit(Job(kind="embed", canonical="cuda-qwen0.6b-1024", items=["y"]))
    assert res == [[1.0]]
    assert ok_calls["n"] == 1


async def test_drop_cancelled_removes_dead_jobs():
    sched, _ = make_sched()
    q = sched._queues.setdefault("cuda-qwen0.6b-1024", collections.deque())
    live = Job(kind="embed", canonical="cuda-qwen0.6b-1024", items=["a"])
    dead = Job(kind="embed", canonical="cuda-qwen0.6b-1024", items=["b"])
    for j in (live, dead):
        j.future = asyncio.get_running_loop().create_future()
    dead.future.cancel()
    q.extend([live, dead])
    sched._drop_cancelled()
    remaining = list(sched._queues["cuda-qwen0.6b-1024"])
    assert remaining == [live]
    assert sched._metrics["cancelled_dropped"] == 1


def test_prometheus_text_shape():
    sched, _ = make_sched()
    sched._metrics["admits"] = 3
    text = sched.prometheus_text()
    assert "gpu_gateway_admits 3" in text
    assert "gpu_gateway_dispatcher_alive" in text
