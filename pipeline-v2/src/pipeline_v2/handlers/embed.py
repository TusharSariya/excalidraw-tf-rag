"""Embed stage handler: chunks artifact -> gpu-gateway ``/v1/embeddings`` -> embeddings artifact.

Talks HTTP to the gpu-gateway's OpenAI-compatible embeddings endpoint
(``gpu_gateway.main.GatewayHTTP``) rather than loading any model in-process —
this stage owns none of the OOM/batching logic, that's fully inside the
gateway per the decision log's "Embed-OOM relocated into gpu-gateway" entry.

Retry semantics for gateway 503
--------------------------------
The gateway only returns 503 after it has *already* exhausted its own
in-process OOM-catch-halve-batch-retry loop (see
``gpu_gateway.main.GatewayHTTP.__call__`` / ``_is_embed_oom``) — by the time
a 503 reaches this handler, the gateway has decided the request cannot be
served *right now* (VRAM pressure), not that the request itself is bad. That
makes 503 (and any 5xx) admission-rejection: retried here with capped,
jittered exponential backoff, and if still failing after the retry budget,
reported as ``HandlerResult(state="failed")`` — which the consumer loop's
existing RELEASE-until-delivery-count-exhausted logic already treats as
retryable (redelivered, then dead-lettered only once the share group's
delivery-count limit is hit). A 4xx (bad request — e.g. unknown model,
malformed input) is *not* retried: it is reported as a distinct
``EmbedPermanentError`` and also surfaces as ``state="failed"`` (the shared
consumer loop has no separate "never retry this" lane — see decision log
Lane A4, DLQ is app-level and threshold-based, not per-message-cause-based),
but the ``reason`` string is tagged ``permanent:`` vs ``retryable:`` so a
human or a future targeted DLQ policy can tell them apart without changing
the loop.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from collections.abc import Sequence

import httpx

from pipeline_v2.artifacts import ArtifactStore, artifact_store_from_env
from pipeline_v2.consumers import HandlerResult
from pipeline_v2.contracts import StageMessage

logger = logging.getLogger(__name__)

DEFAULT_GATEWAY_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL = "cuda-qwen4b-1024"
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_RETRIES = 3
BASE_BACKOFF_S = 0.5
MAX_BACKOFF_S = 5.0


class EmbedRetryableError(RuntimeError):
    """Gateway 503 (admission-rejection) or a transport-level failure."""


class EmbedPermanentError(RuntimeError):
    """Gateway 4xx — the request itself is bad; retrying it won't help."""


def _post_embeddings(
    client: httpx.Client, gateway_url: str, model: str, texts: Sequence[str], timeout: float
) -> dict:
    response = client.post(
        f"{gateway_url.rstrip('/')}/v1/embeddings",
        json={"model": model, "input": list(texts)},
        timeout=timeout,
    )
    if response.status_code == 503:
        raise EmbedRetryableError(f"gateway 503 (admission-rejected): {response.text[:300]}")
    if response.status_code >= 500:
        raise EmbedRetryableError(f"gateway {response.status_code}: {response.text[:300]}")
    if response.status_code >= 400:
        raise EmbedPermanentError(f"gateway {response.status_code}: {response.text[:300]}")
    return response.json()


def _embed_with_retry(
    client: httpx.Client,
    gateway_url: str,
    model: str,
    texts: Sequence[str],
    *,
    max_retries: int,
    timeout: float,
    sleep=time.sleep,
) -> dict:
    attempt = 0
    while True:
        try:
            return _post_embeddings(client, gateway_url, model, texts, timeout)
        except EmbedPermanentError:
            raise
        except (EmbedRetryableError, httpx.TransportError) as exc:
            attempt += 1
            if attempt >= max_retries:
                if isinstance(exc, EmbedRetryableError):
                    raise
                raise EmbedRetryableError(f"transport error after retries: {exc}") from exc
            backoff = min(MAX_BACKOFF_S, BASE_BACKOFF_S * (2 ** (attempt - 1)))
            sleep(backoff + random.uniform(0, backoff * 0.25))


def make_embed_handler(
    store: ArtifactStore | None = None,
    *,
    gateway_url: str | None = None,
    model: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    max_retries: int = DEFAULT_MAX_RETRIES,
    client: httpx.Client | None = None,
):
    """Build the embed stage handler. ``store``/``client`` are injectable for tests."""
    artifact_store = store if store is not None else artifact_store_from_env()
    resolved_url = gateway_url or os.getenv("PIPELINE_GATEWAY_URL", DEFAULT_GATEWAY_URL)
    resolved_model = model or os.getenv("PIPELINE_GATEWAY_EMBED_MODEL", DEFAULT_MODEL)

    def _handler(message: StageMessage) -> HandlerResult:
        return _embed_one(
            message,
            artifact_store,
            gateway_url=resolved_url,
            model=resolved_model,
            timeout=timeout,
            max_retries=max_retries,
            client=client,
        )

    return _handler


def _embed_one(
    message: StageMessage,
    store: ArtifactStore,
    *,
    gateway_url: str,
    model: str,
    timeout: float,
    max_retries: int,
    client: httpx.Client | None,
) -> HandlerResult:
    artifact_uri = message.payload.get("chunks_artifact_uri")
    if not artifact_uri:
        return HandlerResult(state="degraded", reason="payload missing 'chunks_artifact_uri'")

    try:
        raw = store.get(artifact_uri)
        data = json.loads(raw)
    except Exception as exc:  # noqa: BLE001 - artifact read failure is not a gateway concern
        return HandlerResult(state="degraded", reason=f"could not read chunks artifact: {exc}")

    chunks = data.get("chunks") or []
    if not chunks:
        return HandlerResult(state="degraded", reason="chunks artifact has no chunks")

    texts = [str(chunk.get("embed_text") or chunk.get("text") or "") for chunk in chunks]

    owns_client = client is None
    http_client = client if client is not None else httpx.Client()
    try:
        response = _embed_with_retry(
            http_client, gateway_url, model, texts, max_retries=max_retries, timeout=timeout
        )
    except EmbedPermanentError as exc:
        return HandlerResult(state="failed", reason=f"permanent: {exc}")
    except EmbedRetryableError as exc:
        return HandlerResult(state="failed", reason=f"retryable: {exc}")
    finally:
        if owns_client:
            http_client.close()

    embeddings = response.get("data") or []
    vectors_by_index = {item["index"]: item["embedding"] for item in embeddings}
    if len(vectors_by_index) != len(texts):
        return HandlerResult(
            state="degraded",
            reason=(f"gateway returned {len(vectors_by_index)} embeddings for {len(texts)} chunks"),
        )

    embedded_chunks = [{**chunk, "vector": vectors_by_index[i]} for i, chunk in enumerate(chunks)]

    artifact_body = {
        "doc_id": message.doc_id,
        "corpus": message.corpus,
        "content_sha256": message.content_sha256,
        "stage_fingerprint": message.stage_fingerprint,
        "embed_model": model,
        "chunks": embedded_chunks,
    }
    content = json.dumps(artifact_body, sort_keys=True).encode("utf-8")
    embeddings_artifact_uri = store.put(
        content,
        key_parts=["embed", message.corpus, message.doc_id, message.stage_fingerprint],
        suffix=".json",
    )

    return HandlerResult(
        state="ok",
        artifact_uri=embeddings_artifact_uri,
        output_payload={
            "embeddings_artifact_uri": embeddings_artifact_uri,
            "chunk_count": len(embedded_chunks),
        },
    )
