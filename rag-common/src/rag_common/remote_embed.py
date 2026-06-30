"""Remote embed shim — routes embed/rerank calls to the GPU gateway over HTTP.

Uses stdlib urllib only (no new Mac deps). Sync wrapper: RAG tools are sync.
Hard-fails when gateway is unreachable (no silent fallback).

Activated by setting RAG_GPU_GATEWAY_URL in the environment.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class GatewayConnectionError(RuntimeError):
    """Gateway URL is set but unreachable."""


class GatewayError(RuntimeError):
    """Gateway returned an HTTP error status."""


def _gateway_url() -> str:
    return os.environ["RAG_GPU_GATEWAY_URL"].rstrip("/")


def _api_key() -> str:
    return os.environ.get("RAG_GPU_GATEWAY_KEY", "")


def embed_remote_texts(
    texts: list[str],
    *,
    model_id: str,
) -> list[list[float]]:
    """Call /v1/embeddings on the desktop gateway. Returns one vector per text."""
    url = f"{_gateway_url()}/v1/embeddings"
    key = _api_key()
    body = json.dumps({"model": model_id, "input": texts}).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            **({"X-API-Key": key} if key else {}),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        return [item["embedding"] for item in data["data"]]
    except urllib.error.HTTPError as exc:  # must precede URLError (HTTPError subclasses it)
        detail = exc.read().decode("utf-8", errors="replace")
        raise GatewayError(f"Gateway {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise GatewayConnectionError(f"Gateway unreachable at {url}: {exc}") from exc


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
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        results = data["results"]
        scores = [0.0] * len(passages)
        for r in results:
            scores[r["index"]] = float(r["score"])
        return scores
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise GatewayError(f"Gateway {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise GatewayConnectionError(f"Gateway unreachable at {url}: {exc}") from exc
