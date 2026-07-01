"""Scheduler — the single in-process VRAM dispatcher for GatewayHTTP.

This module is deliberately free of `ray` and `torch` imports so its decision
logic can be unit-tested with plain fakes and no GPU (see tests/test_scheduler.py).
GatewayHTTP owns one Scheduler instance and injects the GPU-touching callables
(embed/rerank/chat serve functions, an unload function, and a measured-free-VRAM
probe). Everything here is pure orchestration + bookkeeping.

## Why a single dispatcher

Every request used to independently run `_evict_for()` → acquire lock → model
call → release. Two coroutines could both pass the budget check before either
loaded, breaching the VRAM ceiling (the TOCTOU race). A single background task
that is the *only* code touching `_resident` and eviction removes that window by
construction: eviction and the load that follows are one sequential step with no
await-interleaving from a competing scheduler.

## Throughput over FIFO

Jobs are queued per canonical model. The dispatcher picks the model to serve next
by affinity (already-resident-with-work first) then queue depth, draining a batch
of same-model jobs into a single coalesced inference call. Co-resident models
(e.g. 0.6B embed + reranker) stay loaded together, so the common embed→rerank
sequence never reloads. A fairness floor (STARVE_MS + DRAIN_MAX streak) keeps a
steadily-fed deep queue from starving a shallow one forever.

## Device parameterization

`_resident` is keyed by device (`dict[str, set[str]]`) and the budget is per
device, even though v1 hardcodes a single `cuda:0`. This keeps N-GPU placement a
later *layer* (add a device-selection step before `_admit`) rather than a rewrite.
"""

from __future__ import annotations

import asyncio
import collections
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

log = logging.getLogger("gpu_gateway.scheduler")

# --- Scheduler tuning constants (overridable via Scheduler.__init__ for tests) ---
DRAIN_MAX = 64  # max jobs coalesced into one inference call
DRAIN_TOKEN_BUDGET = 8192  # ~token cap per coalesced batch (bounds activation memory)
STARVE_MS = 5000.0  # a waiter older than this can preempt a long-served model
HEADROOM_MB = 800  # measured-free margin required above a model's static weight
EVICT_POLL_S = 0.05  # backoff when the measured gate is waiting on VRAM to free
MAX_EVICT_ATTEMPTS = 200  # guard: fail the batch rather than spin forever with nothing to evict

LLM_SENTINEL = "__llm__"  # pseudo-model occupying the whole card (Ollama needs it all)
DEFAULT_DEVICE = "cuda:0"


@dataclass
class Job:
    """One unit of GPU work awaiting the dispatcher.

    `items` are the sizeable inputs (texts to embed / passages to rerank); they
    drive the token-bounded drain and coalescing. `payload` carries kind-specific
    extras (the rerank query, the chat body). `future` is resolved by the
    dispatcher; the submitting request coroutine awaits it.
    """

    kind: str  # "embed" | "rerank" | "chat"
    canonical: str  # canonical model name / LLM_SENTINEL
    items: list = field(default_factory=list)
    payload: dict = field(default_factory=dict)
    enqueued_at: float = 0.0
    future: asyncio.Future | None = None
    token_estimate: int = 0

    def __post_init__(self) -> None:
        if not self.token_estimate and self.items:
            # ~4 chars/token is a coarse but stable proxy; only used to bound batch size.
            self.token_estimate = sum(len(str(x)) for x in self.items) // 4


class Scheduler:
    def __init__(
        self,
        *,
        vram_mb: dict[str, int],
        card_mb: int,
        embed_fn: Callable[[str, list[str]], Awaitable[list[list[float]]]],
        rerank_fn: Callable[[str, list[str]], Awaitable[list[float]]],
        chat_fn: Callable[[dict], Awaitable[dict]],
        unload_fn: Callable[[str], Awaitable[bool]],
        measured_free_mb: Callable[[], int],
        clock: Callable[[], float],
        empty_cache: Callable[[], None] | None = None,
        device: str = DEFAULT_DEVICE,
        mode: str = "affinity",
        drain_max: int = DRAIN_MAX,
        drain_token_budget: int = DRAIN_TOKEN_BUDGET,
        starve_ms: float = STARVE_MS,
        headroom_mb: int = HEADROOM_MB,
        evict_poll_s: float = EVICT_POLL_S,
    ) -> None:
        self._vram_mb = vram_mb
        self._card_mb = card_mb
        self._embed_fn = embed_fn
        self._rerank_fn = rerank_fn
        self._chat_fn = chat_fn
        self._unload_fn = unload_fn
        self._measured_free_mb = measured_free_mb
        self._clock = clock
        self._empty_cache = empty_cache or (lambda: None)
        self._device = device
        # "affinity" (default) = bin-packing co-residency + coalesced drain.
        # "naive" = single-resident load-evict-per-request baseline (co-residency
        # off, drain=1); the A/B control curve for the wedge proof (V11). RAM cache
        # is off by config default, so naive needs no handler change.
        self._mode = mode
        self._drain_max = drain_max
        self._drain_token_budget = drain_token_budget
        self._starve_ms = starve_ms
        self._headroom_mb = headroom_mb
        self._evict_poll_s = evict_poll_s

        self._queues: dict[str, collections.deque] = {}
        # Device-parameterized resident set (single device in v1).
        self._resident: dict[str, set[str]] = {device: set()}
        self._parked: set[str] = set()  # evicted-to-RAM (observability only)
        self._loaded_at: dict[str, float] = {}  # canonical → clock() of last serve (LRU)
        # Per-model lifecycle state for well-defined reconcile after failure.
        self._state: dict[str, str] = {}  # "loading"|"loaded"|"parked"|"failed"

        self._last_served: str | None = None
        self._served_streak: int = 0

        self._job_ready: asyncio.Event | None = None
        self._task: asyncio.Task | None = None

        self._metrics = collections.Counter()
        self._last_trace: dict = {}

    # ------------------------------------------------------------------
    # Public API — request handlers call submit(); GatewayHTTP starts the loop.
    # ------------------------------------------------------------------

    async def submit(self, job: Job):
        """Enqueue a job and await its result. Starts the dispatcher lazily."""
        self.ensure_started()
        if job.enqueued_at == 0.0:
            job.enqueued_at = self._clock()
        if job.future is None:
            job.future = asyncio.get_running_loop().create_future()
        self._queues.setdefault(job.canonical, collections.deque()).append(job)
        self._metrics["jobs_submitted"] += 1
        assert self._job_ready is not None
        self._job_ready.set()
        return await job.future

    def ensure_started(self) -> None:
        if self._job_ready is None:
            self._job_ready = asyncio.Event()
        if self._task is None or self._task.done():
            self._task = asyncio.ensure_future(self._run_supervised())

    def alive(self) -> bool:
        return self._task is not None and not self._task.done()

    # ------------------------------------------------------------------
    # Dispatcher loop (supervised) — the ONLY code that mutates _resident.
    # ------------------------------------------------------------------

    async def _run_supervised(self) -> None:
        while True:
            try:
                await self._run()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — the loop scaffolding itself failed
                self._metrics["supervisor_restarts"] += 1
                log.exception("dispatcher loop crashed; restarting: %s", exc)
                await asyncio.sleep(0.1)

    async def _run(self) -> None:
        assert self._job_ready is not None
        while True:
            model: str | None = None
            batch: list[Job] = []
            try:
                if not self._any_queued():
                    await self._job_ready.wait()
                    self._job_ready.clear()
                    continue
                self._drop_cancelled()
                if not self._any_queued():
                    continue
                model = self._select_model()
                if model is None:
                    continue
                batch = self._drain(model)
                if not batch:
                    continue
                await self._admit(model)
                await self._serve(model, batch)
                self._post_serve(model, batch)
            except Exception as exc:  # noqa: BLE001 — error boundary: fail this batch, keep looping
                self._metrics["dispatcher_errors"] += 1
                log.exception("dispatch error for %s (batch=%d): %s", model, len(batch), exc)
                for job in batch:
                    self._set_exception(job, exc)
                await self._reconcile(model)

    # ------------------------------------------------------------------
    # Selection — pure decision logic (unit-tested without a GPU).
    # ------------------------------------------------------------------

    def _select_model(self) -> str | None:
        """Pick the next model to serve: fairness → resident-with-work → deepest queue."""
        starved = self._find_starved()
        if starved is not None:
            self._metrics["starvation_preempts"] += 1
            return starved

        resident = self._resident[self._device]
        working_resident = [m for m in resident if self._qlen(m) > 0]
        if working_resident:
            # Free affinity: serving a resident model costs no load — retire the deepest.
            return max(working_resident, key=self._qlen)

        nonempty = [m for m in self._queues if self._qlen(m) > 0]
        if not nonempty:
            return None
        return max(nonempty, key=self._qlen)

    def _find_starved(self) -> str | None:
        """A non-recently-served queue whose head has waited past STARVE_MS, once the
        recently-served model has had its DRAIN_MAX streak. Bounds worst-case wait."""
        if self._served_streak < self._drain_max:
            return None
        now = self._clock()
        oldest = None
        oldest_wait = self._starve_ms
        for m, q in self._queues.items():
            if not q or m == self._last_served:
                continue
            wait_ms = (now - q[0].enqueued_at) * 1000.0
            if wait_ms > oldest_wait:
                oldest, oldest_wait = m, wait_ms
        return oldest

    def _drain(self, model: str) -> list[Job]:
        """Pull head-of-line jobs for `model`. Coalescable (embed) models drain up to
        DRAIN_MAX jobs or the token budget; rerank/chat drain one at a time."""
        q = self._queues.get(model)
        if not q:
            return []
        # Naive baseline serves one request per load — no coalescing, ever.
        if self._mode == "naive":
            return [q.popleft()]
        if not self._coalescable(model):
            return [q.popleft()]
        batch: list[Job] = []
        toks = 0
        while q and len(batch) < self._drain_max:
            nxt = q[0]
            if batch and toks + nxt.token_estimate > self._drain_token_budget:
                break
            batch.append(q.popleft())
            toks += nxt.token_estimate
        return batch

    # ------------------------------------------------------------------
    # Admission + eviction — bin-packing with a measured-VRAM safety gate.
    # ------------------------------------------------------------------

    async def _admit(self, model: str) -> None:
        resident = self._resident[self._device]
        if model in resident:
            return
        needed = self._effective_mb(model)

        # Naive baseline: never co-reside. Evict every other model first so the
        # target loads alone (the pre-scheduler load-evict-per-request behavior).
        # The measured gate below still runs, so naive never OOMs either.
        if self._mode == "naive":
            for victim in [m for m in list(resident) if m != model]:
                await self._evict(victim)

        # Fast static planner: only evict when the weights budget says we must. For the
        # LLM this evicts every tracked model (needed = whole card).
        if needed > self._static_free():
            await self._evict_for(model, needed)

        # Measured safety gate: static weights ignore CUDA context + activation memory
        # (e.g. 4B+reranker = 7000 static leaves ~200 MB and would OOM). Evict more or
        # wait; never trigger a load we can't fit. The LLM loads its weights
        # out-of-process (Ollama) *after* this admit, so we can't pre-measure its
        # footprint — gate only on freeing our own tracked models (gate_needed = 0);
        # the residency check happens on the NEXT admit instead.
        gate_needed = 0 if model == LLM_SENTINEL else needed
        attempts = 0
        while self._measured_free_mb() < gate_needed + self._headroom_mb:
            if not await self._evict_one_more(model):
                attempts += 1
                if attempts >= MAX_EVICT_ATTEMPTS:
                    raise RuntimeError(
                        f"admit({model}): measured free {self._measured_free_mb()} MB "
                        f"< {gate_needed + self._headroom_mb} MB and nothing left to evict"
                    )
                await asyncio.sleep(self._evict_poll_s)
                continue
            attempts = 0

        # Load/unpark is lazy inside the handler's first serve call — the single
        # dispatcher guarantees one load at a time (Tension 3: never asyncio.gather).
        resident.add(model)
        self._state[model] = "loading"
        self._metrics["admits"] += 1

    async def _evict_for(self, model: str, needed: int) -> None:
        """Free the minimum VRAM so `needed` fits, evicting cheapest-to-lose first."""
        resident = self._resident[self._device]
        while self._card_mb - sum(self._effective_mb(m) for m in resident) < needed:
            victim = self._pick_victim(model)
            if victim is None:
                break  # nothing left; the measured gate will wait/raise
            await self._evict(victim)

    async def _evict_one_more(self, model: str) -> bool:
        victim = self._pick_victim(model)
        if victim is None:
            return False
        await self._evict(victim)
        return True

    def _pick_victim(self, exclude: str) -> str | None:
        """Victim ranking: idle before busy → least-recently-used → shallower queue."""
        cands = [m for m in self._resident[self._device] if m != exclude]
        if not cands:
            return None
        return min(
            cands,
            key=lambda m: (self._qlen(m) > 0, self._loaded_at.get(m, 0.0), self._qlen(m)),
        )

    async def _evict(self, victim: str) -> None:
        parked = False
        try:
            parked = bool(await self._unload_fn(victim))
        except Exception as exc:  # noqa: BLE001 — actor may already be dead
            log.warning("evict %s failed (may already be gone): %s", victim, exc)
        self._resident[self._device].discard(victim)
        self._loaded_at.pop(victim, None)
        if parked:
            self._parked.add(victim)
            self._state[victim] = "parked"
        else:
            self._parked.discard(victim)
            self._state[victim] = "evicted"
        self._empty_cache()
        self._metrics["evicts"] += 1
        log.info("evict: %s (parked=%s)", victim, parked)

    async def _reconcile(self, model: str | None) -> None:
        """After a failure, drop the model from tracking and resync to real VRAM."""
        if model is not None:
            self._resident[self._device].discard(model)
            self._parked.discard(model)
            self._loaded_at.pop(model, None)
            self._state[model] = "failed"
        self._empty_cache()
        self._metrics["reconciles"] += 1

    # ------------------------------------------------------------------
    # Serving — coalesce embed batches; rerank/chat serve per job.
    # ------------------------------------------------------------------

    async def _serve(self, model: str, batch: list[Job]) -> None:
        kind = batch[0].kind
        if kind == "chat":
            for job in batch:
                res = await self._chat_fn(job.payload)
                self._set_result(job, res)
            return
        if kind == "rerank":
            for job in batch:
                try:
                    scores = await self._rerank_fn(job.payload.get("query", ""), job.items)
                    self._set_result(job, scores)
                except Exception as exc:  # noqa: BLE001
                    self._set_exception(job, exc)
            return

        # embed: coalesce all drained jobs into ONE inference call, scatter by span.
        texts: list[str] = []
        spans: list[tuple[int, int]] = []
        for job in batch:
            spans.append((len(texts), len(texts) + len(job.items)))
            texts.extend(job.items)
        try:
            vecs = await self._embed_fn(model, texts)
            for job, (a, b) in zip(batch, spans):
                self._set_result(job, vecs[a:b])
            self._metrics["coalesced_calls"] += 1
        except Exception as exc:  # noqa: BLE001 — one bad payload shouldn't fail all 64
            log.warning("coalesced embed failed; per-job fallback: %s", exc)
            self._metrics["coalesce_fallbacks"] += 1
            for job in batch:
                try:
                    v = await self._embed_fn(model, job.items)
                    self._set_result(job, v)
                except Exception as jexc:  # noqa: BLE001
                    self._set_exception(job, jexc)

    def _post_serve(self, model: str, batch: list[Job]) -> None:
        now = self._clock()
        if model == LLM_SENTINEL:
            # Ollama manages its own VRAM; drop the sentinel so the NEXT admit's
            # measured gate waits for residency to actually clear (Codex #2 / V9).
            self._resident[self._device].discard(model)
            self._state[model] = "evicted"
        else:
            self._loaded_at[model] = now
            self._state[model] = "loaded"
        served = len(batch)
        if model == self._last_served:
            self._served_streak += served
        else:
            self._last_served = model
            self._served_streak = served
        self._metrics["jobs_served"] += served
        self._last_trace = {
            "model": model,
            "batch": served,
            "resident": sorted(self._resident[self._device]),
            "measured_free_mb": self._measured_free_mb(),
            "queue_wait_ms": round((now - batch[0].enqueued_at) * 1000.0, 1),
        }

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _coalescable(self, model: str) -> bool:
        # Only dense embed calls share a homogeneous batched signature.
        q = self._queues.get(model)
        if q:
            return q[0].kind == "embed"
        return model != LLM_SENTINEL and model in self._vram_mb and "reranker" not in model

    def _effective_mb(self, model: str) -> int:
        if model == LLM_SENTINEL:
            return self._card_mb  # LLM needs the whole card
        return self._vram_mb.get(model, 1800)

    def _static_free(self) -> int:
        return self._card_mb - sum(self._effective_mb(m) for m in self._resident[self._device])

    def _qlen(self, model: str) -> int:
        q = self._queues.get(model)
        return len(q) if q else 0

    def _any_queued(self) -> bool:
        return any(q for q in self._queues.values())

    def _drop_cancelled(self) -> None:
        """Purge jobs whose caller disconnected so dead work never triggers a load."""
        for model, q in self._queues.items():
            if not q:
                continue
            live = collections.deque(
                j for j in q if j.future is not None and not j.future.cancelled()
            )
            dropped = len(q) - len(live)
            if dropped:
                self._metrics["cancelled_dropped"] += dropped
                self._queues[model] = live

    @staticmethod
    def _set_result(job: Job, value) -> None:
        if job.future is not None and not job.future.done():
            job.future.set_result(value)

    @staticmethod
    def _set_exception(job: Job, exc: BaseException) -> None:
        if job.future is not None and not job.future.done():
            job.future.set_exception(exc)

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def metrics_snapshot(self) -> dict:
        return {
            "counters": dict(self._metrics),
            "resident": {d: sorted(s) for d, s in self._resident.items()},
            "parked": sorted(self._parked),
            "queue_depth": {m: self._qlen(m) for m in self._queues if self._qlen(m)},
            "dispatcher_alive": self.alive(),
            "measured_free_mb": self._measured_free_mb(),
            "last_trace": self._last_trace,
        }

    def prometheus_text(self) -> str:
        snap = self.metrics_snapshot()
        lines: list[str] = []
        for name, val in snap["counters"].items():
            lines.append(f"gpu_gateway_{name} {val}")
        lines.append(f"gpu_gateway_dispatcher_alive {1 if snap['dispatcher_alive'] else 0}")
        lines.append(f"gpu_gateway_measured_free_mb {snap['measured_free_mb']}")
        lines.append(f"gpu_gateway_resident_count {len(snap['resident'].get(self._device, []))}")
        lines.append(f"gpu_gateway_parked_count {len(snap['parked'])}")
        for m, d in snap["queue_depth"].items():
            lines.append(f'gpu_gateway_queue_depth{{model="{m}"}} {d}')
        return "\n".join(lines) + "\n"
