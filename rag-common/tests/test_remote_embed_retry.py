"""Retry semantics for the gateway HTTP client (remote_embed._request_json).

Added after the 2026-07-03 P0 crash gate: killing the gateway pod mid-embed
left the old no-retry client's consumer thread dead while ingest's main loop
waited forever. The retry window must bridge a pod restart; exhaustion must
raise (fail loud), never hang.
"""

from __future__ import annotations

import io
import json
import urllib.error
from unittest import mock

import pytest
from rag_common import remote_embed
from rag_common.remote_embed import (
    GatewayConnectionError,
    GatewayError,
    _request_json,
)


def _req() -> urllib.request.Request:
    import urllib.request

    return urllib.request.Request("http://gw.test/v1/embeddings", data=b"{}")


def _ok_response(payload: dict):
    resp = mock.MagicMock()
    resp.read.return_value = json.dumps(payload).encode()
    resp.__enter__ = mock.Mock(return_value=resp)
    resp.__exit__ = mock.Mock(return_value=False)
    return resp


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://gw.test", code, "err", hdrs=None, fp=io.BytesIO(b"boom"))


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(remote_embed.time, "sleep", lambda _s: None)


def test_retries_connection_errors_then_succeeds(monkeypatch):
    calls = [
        urllib.error.URLError("refused"),
        TimeoutError("timed out"),
        _ok_response({"data": []}),
    ]

    def fake_urlopen(_req, timeout):
        item = calls.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(remote_embed.urllib.request, "urlopen", fake_urlopen)
    assert _request_json(_req(), timeout=1, url="http://gw.test") == {"data": []}
    assert not calls  # all three attempts consumed


def test_retries_503_then_succeeds(monkeypatch):
    calls = [_http_error(503), _ok_response({"data": [1]})]

    def fake_urlopen(_req, timeout):
        item = calls.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(remote_embed.urllib.request, "urlopen", fake_urlopen)
    assert _request_json(_req(), timeout=1, url="http://gw.test") == {"data": [1]}


def test_permanent_http_error_never_retries(monkeypatch):
    attempts = 0

    def fake_urlopen(_req, timeout):
        nonlocal attempts
        attempts += 1
        raise _http_error(400)

    monkeypatch.setattr(remote_embed.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(GatewayError, match="Gateway 400"):
        _request_json(_req(), timeout=1, url="http://gw.test")
    assert attempts == 1


def test_exhaustion_raises_connection_error(monkeypatch):
    attempts = 0

    def fake_urlopen(_req, timeout):
        nonlocal attempts
        attempts += 1
        raise urllib.error.URLError("still down")

    monkeypatch.setattr(remote_embed.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(GatewayConnectionError, match="after 8 attempts"):
        _request_json(_req(), timeout=1, url="http://gw.test")
    assert attempts == remote_embed._RETRY_MAX_ATTEMPTS
