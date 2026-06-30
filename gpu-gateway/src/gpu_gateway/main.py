"""GatewayHTTP — Ray Serve router with automatic LRU VRAM eviction.

This module's `app` export is what `serve run config/serve_config.yaml` imports
(import_path: gpu_gateway.main:app). It must be a ray.serve.Application built
with serve.bind(), not a bare FastAPI app.

## VRAM management

GatewayHTTP tracks which models are "known to be loaded" in _loaded_at (model →
monotonic timestamp). Before every embed/rerank dispatch it calls _evict_for(),
which evicts LRU models until the new model fits within _CARD_MB. This means:

  - 0.6B ↔ 4B switches are automatic — no manual kill required
  - LLM requests still evict everything (Ollama needs the full card)
  - On actor crash/restart the model is cleared from tracking; the next request
    reloads it after any necessary eviction

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
from starlette.responses import JSONResponse

from gpu_gateway.cache import MODEL_DIMS, SQLiteCache, embed_cache_key, rerank_cache_key
from gpu_gateway.coordinator import VRAMCoordinator
from gpu_gateway.handlers.dense_embed import DenseEmbed06B, DenseEmbed4B
from gpu_gateway.handlers.llm import LLMDeployment
from gpu_gateway.handlers.reranker import RerankerDeployment

log = logging.getLogger("gpu_gateway.main")

# ---------------------------------------------------------------------------
# VRAM budget table (MB) — measured on RTX 3060 Ti with expandable_segments:True.
# Used by _evict_for() to decide what to evict before loading a new model.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Tools root — three levels up from this file (gpu_gateway/ → src/ → gpu-gateway/ → tools/)
# ---------------------------------------------------------------------------
_TOOLS_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
_KNOWN_TOOLS = {"graph-layout-rag", "repo-rag", "rag-literature-rag"}

_VRAM_MB: dict[str, int] = {
    "cuda-qwen4b-1024": 5800,    # Qwen3-4B int8, 1024-dim MRL
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
        # VRAMCoordinator and SQLiteCache are @ray.remote actors (not @serve.deployment).
        # We create them lazily here (get_if_exists via try/except) with lifetime="detached"
        # so they survive GatewayHTTP restarts without namespace-matching issues.
        try:
            self._coordinator = ray.get_actor("gateway_coordinator")
        except ValueError:
            self._coordinator = VRAMCoordinator.options(
                name="gateway_coordinator", lifetime="detached", num_cpus=0
            ).remote()
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

        # LRU VRAM tracker: canonical model name → last-used monotonic timestamp.
        # Set when a model call succeeds; cleared when unloaded or on actor error.
        self._loaded_at: dict[str, float] = {}

    # ------------------------------------------------------------------
    # VRAM orchestration
    # ------------------------------------------------------------------

    def _handle_for(self, canonical: str) -> Any:
        if canonical == "cuda-qwen4b-1024":
            return self._embed4b
        if canonical == "bge-reranker-v2-m3":
            return self._reranker
        return self._embed06b

    async def _evict_for(self, canonical: str) -> None:
        """LRU-evict loaded models until `canonical` fits within _CARD_MB.

        Never evicts the model we're about to load (it may already be loaded
        and we just want a second call — timestamp will be updated after).
        """
        needed = _VRAM_MB.get(canonical, 1800)
        current_total = sum(_VRAM_MB.get(m, 0) for m in self._loaded_at)

        while current_total + needed > _CARD_MB:
            # candidates: everything loaded except the target model
            candidates = [(m, ts) for m, ts in self._loaded_at.items() if m != canonical]
            if not candidates:
                log.warning(
                    "cannot evict further — budget %d MB still needed for %s (%d MB)",
                    _CARD_MB, canonical, needed,
                )
                break
            evict = min(candidates, key=lambda x: x[1])[0]
            evict_mb = _VRAM_MB.get(evict, 0)
            log.info(
                "vram-evict: unloading %s (%d MB) to make room for %s (%d MB)",
                evict, evict_mb, canonical, needed,
            )
            try:
                await self._handle_for(evict).unload.remote()
            except Exception as exc:  # noqa: BLE001
                log.warning("evict %s failed (actor may already be dead): %s", evict, exc)
            self._loaded_at.pop(evict, None)
            current_total -= evict_mb

    async def _evict_all(self) -> None:
        """Unload every tracked model (used before LLM requests)."""
        for canonical in list(self._loaded_at):
            try:
                await self._handle_for(canonical).unload.remote()
            except Exception as exc:  # noqa: BLE001
                log.warning("evict-all %s failed: %s", canonical, exc)
        self._loaded_at.clear()

    def _mark_loaded(self, canonical: str) -> None:
        self._loaded_at[canonical] = time.monotonic()

    def _mark_unloaded(self, canonical: str) -> None:
        self._loaded_at.pop(canonical, None)

    # ------------------------------------------------------------------
    # Request handlers
    # ------------------------------------------------------------------

    async def __call__(self, request: Request) -> JSONResponse:
        path = request.url.path
        if path in ("/health", "/health/"):
            return await self._health()
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
        model = body.get("model", "cuda-qwen4b-1024")
        canonical = _model_handle_name(model)
        texts: list[str] = body.get("input", [])
        if isinstance(texts, str):
            texts = [texts]

        # Cache lookup
        keys = [embed_cache_key(model, t) for t in texts]
        cached = await self._cache.batch_get.remote(keys)
        hits = {k: v for k, v in cached.items()}

        miss_indices = [i for i, k in enumerate(keys) if k not in hits]
        miss_texts = [texts[i] for i in miss_indices]

        if miss_texts:
            handle = self._handle_for(canonical)
            # Evict LRU models if necessary, then acquire the embed slot.
            await self._evict_for(canonical)
            acquired = False
            try:
                await self._coordinator.acquire.remote("embed")
                acquired = True
                new_vecs: list[list[float]] = await handle.remote(miss_texts)
                self._mark_loaded(canonical)
            except Exception:
                # Actor may have crashed/been restarted — clear stale tracking.
                self._mark_unloaded(canonical)
                raise
            finally:
                if acquired:
                    await self._coordinator.release.remote("embed")

            store_entries = [
                (keys[miss_indices[j]], model, new_vecs[j])
                for j in range(len(miss_texts))
            ]
            await self._cache.batch_set.remote(store_entries)
            for j, idx in enumerate(miss_indices):
                hits[keys[idx]] = new_vecs[j]

        x_cache = "MISS" if miss_texts else "HIT"
        data = [
            {"index": i, "embedding": hits[k], "object": "embedding"}
            for i, k in enumerate(keys)
        ]
        return JSONResponse(
            {"object": "list", "data": data, "model": model},
            headers={"X-Cache": x_cache},
        )

    async def _rerank(self, body: dict) -> JSONResponse:
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
            await self._evict_for(canonical)
            acquired = False
            try:
                await self._coordinator.acquire.remote("embed")
                acquired = True
                scores: list[float] = await self._reranker.remote(query, miss_passages)
                self._mark_loaded(canonical)
            except Exception:
                self._mark_unloaded(canonical)
                raise
            finally:
                if acquired:
                    await self._coordinator.release.remote("embed")

            store = [
                (keys[miss_indices[j]], model, [scores[j]])
                for j in range(len(miss_passages))
            ]
            await self._cache.batch_set.remote(store)
            for j, idx in enumerate(miss_indices):
                cached[keys[idx]] = [scores[j]]

        results = [
            {"index": i, "score": round(cached[k][0], 6)}
            for i, k in enumerate(keys)
        ]
        return JSONResponse({"results": results, "model": model})

    async def _chat(self, body: dict) -> JSONResponse:
        # Ollama needs the full card — evict everything tracked.
        await self._evict_all()
        acquired = False
        try:
            await self._coordinator.acquire.remote("llm")
            acquired = True
            result = await self._llm.remote(body)
        finally:
            if acquired:
                await self._coordinator.release.remote("llm")
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
            return JSONResponse(
                {"error": f"tool root not found: {tool_root}"}, status_code=500
            )

        uv = shutil.which("uv") or os.path.expanduser("~/.local/bin/uv")
        cmd = [
            uv, "run", "--no-sync", tool,
            "query", query,
            "--embed-profile", profile,
            "--top", str(top),
            "--json",
            *flags,
        ]

        env = os.environ.copy()
        env["RAG_GPU_GATEWAY_URL"] = "http://localhost:8765"
        env.pop("GRAPH_RAG_GATEWAY_SEARCH", None)   # prevent recursion
        env.pop("RAG_LOCAL_EMBED_DEVICE", None)      # never load models directly

        log.info("search: tool=%s profile=%s top=%d query=%r flags=%s",
                 tool, profile, top, query[:80], flags)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=tool_root,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120.0)
        except asyncio.TimeoutError:
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
                ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                timeout=5,
            ).decode()
            for i, line in enumerate(out.strip().splitlines()):
                used, total = line.split(",")
                gpu_info.append({
                    "device": f"cuda:{i}",
                    "used_mb": int(used.strip()),
                    "total_mb": int(total.strip()),
                })
        except Exception:  # noqa: BLE001
            pass

        cache_count = await self._cache.count.remote()
        return JSONResponse({
            "status": "ok",
            "gpus": gpu_info,
            "cache_entries": cache_count,
            # expose tracked model state for observability
            "loaded_models": {
                m: round(time.monotonic() - ts, 1)
                for m, ts in self._loaded_at.items()
            },
        })


# serve.bind() wiring — exported as `app` for `serve run config/serve_config.yaml`
# VRAMCoordinator and SQLiteCache are @ray.remote actors created inside GatewayHTTP.__init__.
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
