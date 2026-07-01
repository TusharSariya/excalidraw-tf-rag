"""GatewayHTTP — Ray Serve router with a single in-process VRAM dispatcher.

This module's `app` export is what `serve run config/serve_config.yaml` imports
(import_path: gpu_gateway.main:app). It must be a ray.serve.Application built
with serve.bind(), not a bare FastAPI app.

## VRAM management

All GPU work funnels through one `Scheduler` (see scheduler.py). Request handlers
enqueue a `Job` and await its future; a single background dispatcher task is the
*only* code that touches VRAM, evicts, and loads — so eviction and the load that
follows are one sequential step (no TOCTOU race). The dispatcher bin-packs
co-resident models (0.6B embed + reranker stay loaded together), schedules by
queue depth, and guards every load with a measured-VRAM gate. This means:

  - 0.6B ↔ 4B switches are automatic — no manual kill required
  - embed→rerank no longer thrashes (both fit at 3000 MB, stay co-resident)
  - LLM requests still evict everything (Ollama needs the full card)
  - a load failure fails only that batch; the supervised loop keeps running

VRAM budgets are measured observations on the RTX 3060 Ti (8 GB / 7822 MB
usable). Update _VRAM_MB if you load different quantizations.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from typing import Any

import ray
from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse

from gpu_gateway.cache import SQLiteCache, embed_cache_key, rerank_cache_key
from gpu_gateway.handlers.dense_embed import DenseEmbed06B, DenseEmbed4B
from gpu_gateway.handlers.llm import LLMDeployment
from gpu_gateway.handlers.reranker import RerankerDeployment
from gpu_gateway.scheduler import LLM_SENTINEL, Job, Scheduler

log = logging.getLogger("gpu_gateway.main")

# ---------------------------------------------------------------------------
# VRAM budget table (MB) — measured on RTX 3060 Ti with expandable_segments:True.
# The static planner the Scheduler uses to decide bin-packing and eviction; the
# measured-VRAM gate (nvidia-smi) is the runtime safety check on top of it.
# ---------------------------------------------------------------------------
# Tools root — three levels up from this file (gpu_gateway/ → src/ → gpu-gateway/ → tools/)
# ---------------------------------------------------------------------------
_TOOLS_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
_KNOWN_TOOLS = {"graph-layout-rag", "repo-rag", "rag-literature-rag"}

_VRAM_MB: dict[str, int] = {
    "cuda-qwen4b-1024": 5800,  # Qwen3-4B int8, 1024-dim MRL
    "cuda-qwen0.6b-1024": 1800,  # Qwen3-0.6B BF16, seq=512
    "bge-reranker-v2-m3": 1200,  # BGE reranker v2 m3
}
_CARD_MB = 7200  # conservative ceiling — leaves ~600 MB for driver/CUDA overhead


def _model_handle_name(model: str) -> str:
    """Canonical model name for tracking (normalises unknown 0.6B profiles)."""
    if model == "cuda-qwen4b-1024":
        return "cuda-qwen4b-1024"
    if model == "bge-reranker-v2-m3":
        return "bge-reranker-v2-m3"
    return "cuda-qwen0.6b-1024"


@serve.deployment
class GatewayHTTP:
    def __init__(
        self,
        embed4b: Any,
        embed06b: Any,
        reranker: Any,
        llm: Any,
    ) -> None:
        # SQLiteCache is a detached @ray.remote actor (survives GatewayHTTP restarts).
        try:
            self._cache = ray.get_actor("gateway_cache")
        except ValueError:
            self._cache = SQLiteCache.options(
                name="gateway_cache", lifetime="detached", num_cpus=0
            ).remote()
        self._embed4b = embed4b
        self._embed06b = embed06b
        self._reranker = reranker
        self._llm = llm

        # Short-TTL cache for the nvidia-smi free-VRAM probe — the admit gate polls it
        # tightly under pressure, and shelling out each time would block the loop.
        self._free_cache: tuple[float, int] = (0.0, 0)
        self._replica_checked = False

        # The single VRAM dispatcher. It owns all resident/eviction state; handlers
        # just enqueue jobs and await. GPU-touching work is injected as callables so
        # the scheduler stays ray/torch-free and unit-testable.
        # GATEWAY_SCHED_MODE=naive forces the single-resident load-evict baseline
        # (co-residency off, drain=1) used as the A/B control curve for the wedge
        # proof (see scripts/eval_harness.py). Default "affinity" is the real thing.
        self._sched_mode = os.environ.get("GATEWAY_SCHED_MODE", "affinity")
        self._sched = Scheduler(
            vram_mb=_VRAM_MB,
            card_mb=_CARD_MB,
            embed_fn=self._embed_call,
            rerank_fn=self._rerank_call,
            chat_fn=self._chat_call,
            unload_fn=self._unload_call,
            measured_free_mb=self._measured_free_mb,
            clock=time.monotonic,
            mode=self._sched_mode,
        )

    # ------------------------------------------------------------------
    # GPU-touching callables injected into the Scheduler
    # ------------------------------------------------------------------

    def _handle_for(self, canonical: str) -> Any:
        if canonical == "cuda-qwen4b-1024":
            return self._embed4b
        if canonical == "bge-reranker-v2-m3":
            return self._reranker
        return self._embed06b

    async def _embed_call(self, canonical: str, texts: list[str]) -> list[list[float]]:
        return await self._handle_for(canonical).remote(texts)

    async def _rerank_call(self, query: str, passages: list[str]) -> list[float]:
        return await self._reranker.remote(query, passages)

    async def _chat_call(self, payload: dict) -> dict:
        return await self._llm.remote(payload)

    async def _unload_call(self, canonical: str) -> bool:
        """Evict a handler's model; returns True if it was parked in RAM (vs destroyed)."""
        return bool(await self._handle_for(canonical).unload.remote())

    def _measured_free_mb(self) -> int:
        """Real free VRAM on the card (all processes), via nvidia-smi. GatewayHTTP runs
        with num_gpus=0 and has no CUDA context, so torch.cuda.mem_get_info is unavailable
        here; nvidia-smi also correctly counts Ollama's out-of-process usage. 0.25s TTL."""
        now = time.monotonic()
        ts, val = self._free_cache
        if now - ts < 0.25 and ts:
            return val
        free = _CARD_MB
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                timeout=5,
            ).decode()
            free = int(out.strip().splitlines()[0].strip())
        except Exception:  # noqa: BLE001 — assume worst-case-safe full card if probe fails
            pass
        self._free_cache = (now, free)
        return free

    def _assert_single_replica(self) -> None:
        """Fail closed if GatewayHTTP is ever scaled past one replica — two dispatchers
        would split-brain the VRAM view. Best-effort via serve.status(); config also pins
        num_replicas: 1. Checked once, lazily, after the deployment is fully up."""
        if self._replica_checked:
            return
        self._replica_checked = True
        try:
            status = serve.status()
            for app in status.applications.values():
                dep = app.deployments.get("GatewayHTTP")
                if dep is None:
                    continue
                running = sum(
                    n
                    for st, n in getattr(dep, "replica_states", {}).items()
                    if str(st).upper().endswith("RUNNING")
                )
                if running > 1:
                    raise RuntimeError(
                        f"GatewayHTTP has {running} replicas — the in-process VRAM "
                        "dispatcher requires exactly 1 (set num_replicas: 1)."
                    )
        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001 — status API shape varies; config pins 1
            log.debug("replica-count check skipped: %s", exc)

    # ------------------------------------------------------------------
    # Request handlers
    # ------------------------------------------------------------------

    async def __call__(self, request: Request):
        path = request.url.path
        if path in ("/health", "/health/"):
            return await self._health()
        if path in ("/metrics", "/metrics/"):
            return PlainTextResponse(self._sched.prometheus_text())
        if path in ("/v1/embeddings", "/v1/embeddings/"):
            body = await request.json()
            return await self._embeddings(body)
        if path in ("/v1/rerank", "/v1/rerank/"):
            body = await request.json()
            return await self._rerank(body)
        if path in ("/v1/chat/completions", "/v1/chat/completions/"):
            body = await request.json()
            return await self._chat(body)
        if path in ("/v1/search", "/v1/search/"):
            body = await request.json()
            return await self._search(body)
        return JSONResponse({"error": f"unknown path {path}"}, status_code=404)

    async def _embeddings(self, body: dict) -> JSONResponse:
        self._assert_single_replica()
        model = body.get("model", "cuda-qwen4b-1024")
        canonical = _model_handle_name(model)
        texts: list[str] = body.get("input", [])
        if isinstance(texts, str):
            texts = [texts]

        # Cache lookup — hits never touch the GPU (stay out of the dispatcher).
        keys = [embed_cache_key(model, t) for t in texts]
        cached = await self._cache.batch_get.remote(keys)
        hits = {k: v for k, v in cached.items()}

        miss_indices = [i for i, k in enumerate(keys) if k not in hits]
        miss_texts = [texts[i] for i in miss_indices]

        if miss_texts:
            # Enqueue the miss batch and await the dispatcher; it coalesces same-model
            # jobs, bin-packs VRAM, and returns one vector per input text in order.
            new_vecs: list[list[float]] = await self._sched.submit(
                Job(kind="embed", canonical=canonical, items=miss_texts)
            )
            store_entries = [
                (keys[miss_indices[j]], model, new_vecs[j]) for j in range(len(miss_texts))
            ]
            await self._cache.batch_set.remote(store_entries)
            for j, idx in enumerate(miss_indices):
                hits[keys[idx]] = new_vecs[j]

        x_cache = "MISS" if miss_texts else "HIT"
        data = [
            {"index": i, "embedding": hits[k], "object": "embedding"} for i, k in enumerate(keys)
        ]
        return JSONResponse(
            {"object": "list", "data": data, "model": model},
            headers={"X-Cache": x_cache},
        )

    async def _rerank(self, body: dict) -> JSONResponse:
        self._assert_single_replica()
        model = body.get("model", "bge-reranker-v2-m3")
        canonical = "bge-reranker-v2-m3"
        query: str = body.get("query", "")
        passages: list[str] = body.get("passages", [])

        # Cache lookup per passage
        keys = [rerank_cache_key(model, query, p) for p in passages]
        cached = await self._cache.batch_get.remote(keys)

        miss_indices = [i for i, k in enumerate(keys) if k not in cached]
        miss_passages = [passages[i] for i in miss_indices]

        if miss_passages:
            scores: list[float] = await self._sched.submit(
                Job(
                    kind="rerank",
                    canonical=canonical,
                    items=miss_passages,
                    payload={"query": query},
                )
            )
            store = [(keys[miss_indices[j]], model, [scores[j]]) for j in range(len(miss_passages))]
            await self._cache.batch_set.remote(store)
            for j, idx in enumerate(miss_indices):
                cached[keys[idx]] = [scores[j]]

        results = [{"index": i, "score": round(cached[k][0], 6)} for i, k in enumerate(keys)]
        return JSONResponse({"results": results, "model": model})

    async def _chat(self, body: dict) -> JSONResponse:
        # LLM_SENTINEL occupies the whole card: the dispatcher evicts every resident
        # model, serves the LLM alone, then drops the sentinel so the next embed/rerank
        # admit waits (via the measured gate) for Ollama's VRAM to actually clear.
        self._assert_single_replica()
        result = await self._sched.submit(Job(kind="chat", canonical=LLM_SENTINEL, payload=body))
        return JSONResponse(result)

    async def _search(self, body: dict) -> JSONResponse:
        """Run a full RAG search on the desktop, returning JSON results.

        The subprocess runs `uv run <tool> query` in the tool's directory with
        RAG_GPU_GATEWAY_URL=http://localhost:8765 so embedding calls loop back
        through the gateway's own model deployments. GRAPH_RAG_GATEWAY_SEARCH
        is stripped from the subprocess env to prevent infinite recursion.

        Request body:
            tool    — "graph-layout-rag" | "repo-rag" | "rag-literature-rag"
            query   — query text (required)
            profile — embed profile (default: cuda-qwen0.6b-1024)
            top     — number of results (default: 8)
            flags   — extra CLI flags e.g. ["--tag", "constraints", "--pdf-only"]
        """
        import asyncio
        import shutil

        tool = body.get("tool", "graph-layout-rag")
        query = body.get("query", "")
        profile = body.get("profile", "cuda-qwen0.6b-1024")
        top = int(body.get("top", 8))
        flags: list[str] = [str(f) for f in body.get("flags", [])]

        if not query:
            return JSONResponse({"error": "query is required"}, status_code=400)
        if tool not in _KNOWN_TOOLS:
            return JSONResponse({"error": f"unknown tool {tool!r}"}, status_code=400)

        tool_root = os.path.join(_TOOLS_ROOT, tool)
        if not os.path.isdir(tool_root):
            return JSONResponse({"error": f"tool root not found: {tool_root}"}, status_code=500)

        uv = shutil.which("uv") or os.path.expanduser("~/.local/bin/uv")
        cmd = [
            uv,
            "run",
            "--no-sync",
            tool,
            "query",
            query,
            "--embed-profile",
            profile,
            "--top",
            str(top),
            "--json",
            *flags,
        ]

        env = os.environ.copy()
        env["RAG_GPU_GATEWAY_URL"] = "http://localhost:8765"
        env.pop("GRAPH_RAG_GATEWAY_SEARCH", None)  # prevent recursion
        env.pop("REPO_RAG_GATEWAY_SEARCH", None)  # prevent recursion
        env.pop("RAG_LIT_GATEWAY_SEARCH", None)  # prevent recursion
        env.pop("RAG_LOCAL_EMBED_DEVICE", None)  # never load models directly

        log.info(
            "search: tool=%s profile=%s top=%d query=%r flags=%s",
            tool,
            profile,
            top,
            query[:80],
            flags,
        )
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=tool_root,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120.0)
        except TimeoutError:
            return JSONResponse({"error": "search timed out (120s)"}, status_code=504)
        except Exception as exc:  # noqa: BLE001
            log.error("search subprocess error: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=500)

        if proc.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace")
            log.error("search failed exit=%d stderr=%s", proc.returncode, detail[:500])
            return JSONResponse(
                {"error": "search subprocess failed", "detail": detail},
                status_code=500,
            )

        try:
            data = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raw = stdout.decode("utf-8", errors="replace")[:300]
            log.error("search bad JSON: %s | raw=%s", exc, raw)
            return JSONResponse({"error": "bad JSON from search", "raw": raw}, status_code=500)

        return JSONResponse(data)

    async def _health(self) -> JSONResponse:
        gpu_info: list[dict] = []
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                timeout=5,
            ).decode()
            for i, line in enumerate(out.strip().splitlines()):
                used, total = line.split(",")
                gpu_info.append(
                    {
                        "device": f"cuda:{i}",
                        "used_mb": int(used.strip()),
                        "total_mb": int(total.strip()),
                    }
                )
        except Exception:  # noqa: BLE001
            pass

        cache_count = await self._cache.count.remote()
        snap = self._sched.metrics_snapshot()
        # 4B RAM-cache health (validate+fallback may have disabled it) — best-effort.
        ram_cache_4b = "unknown"
        try:
            ram_cache_4b = await self._embed4b.ram_cache_status.remote()
        except Exception:  # noqa: BLE001
            pass
        return JSONResponse(
            {
                "status": "ok",
                "sched_mode": self._sched_mode,
                "gpus": gpu_info,
                "cache_entries": cache_count,
                # Scheduler-tracked model state for observability.
                "loaded_models": snap["resident"].get("cuda:0", []),
                "parked_models": snap["parked"],
                "queue_depth": snap["queue_depth"],
                "dispatcher_alive": snap["dispatcher_alive"],
                "scheduler_counters": snap["counters"],
                "ram_cache_4b": ram_cache_4b,
            }
        )


# serve.bind() wiring — exported as `app` for `serve run config/serve_config.yaml`
# SQLiteCache is a detached @ray.remote actor created inside GatewayHTTP.__init__.
embed4b = DenseEmbed4B.bind()
embed06b = DenseEmbed06B.bind()
reranker = RerankerDeployment.bind()
llm = LLMDeployment.bind()

app = GatewayHTTP.bind(
    embed4b=embed4b,
    embed06b=embed06b,
    reranker=reranker,
    llm=llm,
)
