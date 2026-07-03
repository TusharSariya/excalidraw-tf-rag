"""Remote embed shim — routes embed/rerank calls to the GPU gateway over HTTP.

Uses stdlib urllib only (no new Mac deps). Sync wrapper: RAG tools are sync.
Hard-fails when gateway is unreachable (no silent fallback).

Activated by setting RAG_GPU_GATEWAY_URL in the environment.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request

log = logging.getLogger("rag_common.remote_embed")


class GatewayConnectionError(RuntimeError):
    """Gateway URL is set but unreachable."""


class GatewayError(RuntimeError):
    """Gateway returned an HTTP error status."""


def _gateway_url() -> str:
    return os.environ["RAG_GPU_GATEWAY_URL"].rstrip("/")


def _api_key() -> str:
    return os.environ.get("RAG_GPU_GATEWAY_KEY", "")


_GATEWAY_BATCH = 256  # texts per HTTP call — keeps each call <10 s and VRAM safe

# Retry window must comfortably outlive a gateway pod restart (~2-3 min on
# k3s: reschedule + Ray boot + lazy model reload). Without this, one killed
# pod turns into a dead embed consumer while ingest's main loop waits forever
# — reproduced live by the P0 crash gate on 2026-07-03.
_RETRY_MAX_ATTEMPTS = int(os.environ.get("RAG_GATEWAY_RETRY_ATTEMPTS", "8"))
_RETRY_BACKOFF_CAP_S = 60.0
# 502/503/504: the gateway signals transient capacity (admission rejection
# after its own OOM-halve-retry bottoms out) or a proxy-side blip — retry.
_RETRYABLE_HTTP = {502, 503, 504}


def _request_json(req: urllib.request.Request, *, timeout: float, url: str) -> dict:
    """One HTTP call with bounded exponential-backoff retries on transient
    failures (connection refused/reset, timeouts, 502/503/504). Raises
    GatewayError on permanent HTTP errors, GatewayConnectionError once the
    retry budget is exhausted — callers fail loudly, they never hang."""
    last_exc: Exception | None = None
    for attempt in range(_RETRY_MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code not in _RETRYABLE_HTTP:
                detail = exc.read().decode("utf-8", errors="replace")
                raise GatewayError(f"Gateway {exc.code}: {detail}") from exc
            last_exc = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            # urlopen raises bare TimeoutError on read timeouts and URLError
            # (wrapping the socket error) on connect failures — both transient.
            last_exc = exc
        if attempt < _RETRY_MAX_ATTEMPTS - 1:
            delay = min(2.0**attempt, _RETRY_BACKOFF_CAP_S)
            log.warning(
                "gateway request failed (%s); retry %d/%d in %.0fs",
                last_exc,
                attempt + 1,
                _RETRY_MAX_ATTEMPTS - 1,
                delay,
            )
            time.sleep(delay)
    raise GatewayConnectionError(
        f"Gateway unreachable at {url} after {_RETRY_MAX_ATTEMPTS} attempts: {last_exc}"
    ) from last_exc


def _embed_one_batch(
    texts: list[str],
    *,
    url: str,
    key: str,
    model_id: str,
) -> list[list[float]]:
    body = json.dumps({"model": model_id, "input": texts}).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            **({"X-API-Key": key} if key else {}),
        },
    )
    data = _request_json(req, timeout=60, url=url)
    return [item["embedding"] for item in data["data"]]


def embed_remote_texts(
    texts: list[str],
    *,
    model_id: str,
) -> list[list[float]]:
    """Call /v1/embeddings on the desktop gateway. Returns one vector per text.

    Automatically chunks large inputs into _GATEWAY_BATCH-sized HTTP requests so
    a single ingest call with thousands of chunks doesn't OOM the 8 GB GPU or
    exceed the per-request timeout.
    """
    url = f"{_gateway_url()}/v1/embeddings"
    key = _api_key()
    if len(texts) <= _GATEWAY_BATCH:
        return _embed_one_batch(texts, url=url, key=key, model_id=model_id)
    results: list[list[float]] = []
    for i in range(0, len(texts), _GATEWAY_BATCH):
        results.extend(
            _embed_one_batch(texts[i : i + _GATEWAY_BATCH], url=url, key=key, model_id=model_id)
        )
    return results


def rerank_remote(
    query: str,
    passages: list[str],
    *,
    model_id: str = "bge-reranker-v2-m3",
) -> list[float]:
    """Call /v1/rerank on the desktop gateway. Returns one score per passage."""
    url = f"{_gateway_url()}/v1/rerank"
    key = _api_key()
    body = json.dumps({"model": model_id, "query": query, "passages": passages}).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            **({"X-API-Key": key} if key else {}),
        },
    )
    data = _request_json(req, timeout=30, url=url)
    results = data["results"]
    scores = [0.0] * len(passages)
    for r in results:
        scores[r["index"]] = float(r["score"])
    return scores
