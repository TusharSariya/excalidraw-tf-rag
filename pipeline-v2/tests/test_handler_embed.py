from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from pipeline_v2.artifacts import LocalArtifactStore
from pipeline_v2.contracts import StageMessage
from pipeline_v2.handlers import embed as embed_mod
from pipeline_v2.handlers.embed import make_embed_handler


class _StubGatewayServer:
    """A tiny stdlib HTTP server standing in for gpu-gateway's /v1/embeddings.

    ``script`` is a list of status codes to return for successive requests
    (cycling through 503s before an eventual 200, or a permanent 4xx, etc.).
    """

    def __init__(self, script: list[int], *, dims: int = 4) -> None:
        self.script = list(script)
        self.dims = dims
        self.requests: list[dict] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # noqa: A003 - silence stdlib logging
                pass

            def do_POST(self) -> None:  # noqa: N802 - stdlib naming
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                body = json.loads(raw)
                server.requests.append(body)
                status = server.script.pop(0) if server.script else 200
                if status == 200:
                    texts = body["input"]
                    payload = {
                        "object": "list",
                        "model": body.get("model"),
                        "data": [
                            {"index": i, "object": "embedding", "embedding": [0.1] * server.dims}
                            for i in range(len(texts))
                        ],
                    }
                    data = json.dumps(payload).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)
                elif status == 503:
                    data = json.dumps({"error": "embedding failed: out of GPU memory"}).encode()
                    self.send_response(503)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    data = json.dumps({"error": "bad request"}).encode()
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(data)

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_port}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def stub_gateway():
    servers: list[_StubGatewayServer] = []

    def _make(script: list[int], **kwargs) -> _StubGatewayServer:
        server = _StubGatewayServer(script, **kwargs)
        servers.append(server)
        return server

    yield _make
    for server in servers:
        server.close()


def _chunks_artifact(store, *, n=2) -> str:
    body = {
        "doc_id": "doc-1",
        "corpus": "test-corpus",
        "content_sha256": "sha",
        "stage_fingerprint": "fp",
        "chunks": [
            {
                "doc_id": "doc-1",
                "chunk_index": i,
                "title": "T",
                "text": f"chunk {i}",
                "embed_text": f"chunk {i}",
            }
            for i in range(n)
        ],
    }
    return store.put(
        json.dumps(body).encode(),
        key_parts=["extract", "test-corpus", "doc-1", "fp"],
        suffix=".json",
    )


def _message(artifact_uri: str | None) -> StageMessage:
    payload = {"chunks_artifact_uri": artifact_uri} if artifact_uri else {}
    return StageMessage(
        doc_id="doc-1",
        corpus="test-corpus",
        content_sha256="sha",
        stage_fingerprint="fp",
        payload=payload,
    )


def test_embed_ok_writes_embeddings_artifact(tmp_path, stub_gateway):
    store = LocalArtifactStore(root=tmp_path / "artifacts")
    server = stub_gateway([200])
    handler = make_embed_handler(store=store, gateway_url=server.url, max_retries=3, timeout=5.0)

    artifact_uri = _chunks_artifact(store)
    result = handler(_message(artifact_uri))

    assert result.state == "ok", result.reason
    body = json.loads(store.get(result.artifact_uri))
    assert len(body["chunks"]) == 2
    assert body["chunks"][0]["vector"] == [0.1] * 4


def test_embed_missing_artifact_payload_is_degraded(tmp_path, stub_gateway):
    store = LocalArtifactStore(root=tmp_path / "artifacts")
    server = stub_gateway([200])
    handler = make_embed_handler(store=store, gateway_url=server.url)

    result = handler(_message(None))

    assert result.state == "degraded"
    assert "chunks_artifact_uri" in (result.reason or "")


def test_embed_503_then_ok_retries_and_succeeds(tmp_path, stub_gateway, monkeypatch):
    store = LocalArtifactStore(root=tmp_path / "artifacts")
    server = stub_gateway([503, 503, 200])
    monkeypatch.setattr(embed_mod.time, "sleep", lambda _s: None)  # no real sleeping in tests
    handler = make_embed_handler(store=store, gateway_url=server.url, max_retries=5, timeout=5.0)

    artifact_uri = _chunks_artifact(store)
    result = handler(_message(artifact_uri))

    assert result.state == "ok", result.reason
    assert len(server.requests) == 3


def test_embed_503_exhausts_retries_is_failed_retryable(tmp_path, stub_gateway, monkeypatch):
    store = LocalArtifactStore(root=tmp_path / "artifacts")
    server = stub_gateway([503, 503, 503, 503, 503])
    monkeypatch.setattr(embed_mod.time, "sleep", lambda _s: None)
    handler = make_embed_handler(store=store, gateway_url=server.url, max_retries=3, timeout=5.0)

    artifact_uri = _chunks_artifact(store)
    result = handler(_message(artifact_uri))

    assert result.state == "failed"
    assert result.reason.startswith("retryable:")
    assert len(server.requests) == 3  # bounded by max_retries, no unbounded retry storm


def test_embed_permanent_4xx_is_failed_permanent_no_retry(tmp_path, stub_gateway, monkeypatch):
    store = LocalArtifactStore(root=tmp_path / "artifacts")
    server = stub_gateway([400])
    monkeypatch.setattr(embed_mod.time, "sleep", lambda _s: None)
    handler = make_embed_handler(store=store, gateway_url=server.url, max_retries=5, timeout=5.0)

    artifact_uri = _chunks_artifact(store)
    result = handler(_message(artifact_uri))

    assert result.state == "failed"
    assert result.reason.startswith("permanent:")
    assert len(server.requests) == 1  # not retried
