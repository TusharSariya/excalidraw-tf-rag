"""GatewayHTTP — Ray Serve router.

This module's `app` export is what `serve run config/serve_config.yaml` imports
(import_path: gpu_gateway.main:app). It must be a ray.serve.Application built
with serve.bind(), not a bare FastAPI app.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
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


@serve.deployment
class GatewayHTTP:
    def __init__(
        self,
        coordinator: Any,
        cache: Any,
        embed4b: Any,
        embed06b: Any,
        reranker: Any,
        llm: Any,
    ) -> None:
        self._coordinator = coordinator
        self._cache = cache
        self._embed4b = embed4b
        self._embed06b = embed06b
        self._reranker = reranker
        self._llm = llm

    async def __call__(self, request: Request) -> JSONResponse:
        path = request.url.path
        if path == "/health" or path == "/health/":
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
        return JSONResponse({"error": f"unknown path {path}"}, status_code=404)

    async def _embeddings(self, body: dict) -> JSONResponse:
        model = body.get("model", "cuda-qwen4b-1024")
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
            handle = self._embed4b if model == "cuda-qwen4b-1024" else self._embed06b
            acquired = False
            try:
                await self._coordinator.acquire.remote("embed")
                acquired = True
                new_vecs: list[list[float]] = await handle.remote(miss_texts)
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
        query: str = body.get("query", "")
        passages: list[str] = body.get("passages", [])

        # Cache lookup per passage
        keys = [rerank_cache_key(model, query, p) for p in passages]
        cached = await self._cache.batch_get.remote(keys)

        miss_indices = [i for i, k in enumerate(keys) if k not in cached]
        miss_passages = [passages[i] for i in miss_indices]

        if miss_passages:
            acquired = False
            try:
                await self._coordinator.acquire.remote("embed")
                acquired = True
                scores: list[float] = await self._reranker.remote(query, miss_passages)
            finally:
                if acquired:
                    await self._coordinator.release.remote("embed")

            # Store scores as 1-float vectors in cache
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
        acquired = False
        try:
            await self._coordinator.acquire.remote("llm")
            acquired = True
            result = await self._llm.remote(body)
        finally:
            if acquired:
                await self._coordinator.release.remote("llm")
        return JSONResponse(result)

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
        })


# serve.bind() wiring — exported as `app` for `serve run config/serve_config.yaml`
coordinator = VRAMCoordinator.bind()
cache = SQLiteCache.bind()
embed4b = DenseEmbed4B.bind()
embed06b = DenseEmbed06B.bind()
reranker = RerankerDeployment.bind()
llm = LLMDeployment.bind()

app = GatewayHTTP.bind(
    coordinator=coordinator,
    cache=cache,
    embed4b=embed4b,
    embed06b=embed06b,
    reranker=reranker,
    llm=llm,
)
