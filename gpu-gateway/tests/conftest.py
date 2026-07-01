"""Pytest wiring for the two test tiers.

Tier 0 (test_scheduler.py) is pure logic and runs anywhere — no options needed.
Tier 1 (test_integration.py) drives a *live* gateway and is gated behind
`--gateway-url`. Without that option the integration tests are auto-skipped, so a
plain `pytest tests/` on a GPU-less laptop runs only the unit suite.

    pytest tests/                                   # unit only (integration skipped)
    pytest -m integration --gateway-url http://localhost:8765   # + live checks
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import pytest


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--gateway-url",
        action="store",
        default=None,
        help="Base URL of a live gateway; enables the -m integration checks.",
    )


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers", "integration: requires a live gateway (pass --gateway-url)."
    )


def pytest_collection_modifyitems(config, items) -> None:
    """Auto-skip integration tests unless a gateway URL was supplied, so the
    default `pytest tests/` stays GPU-free and green on the laptop."""
    if config.getoption("--gateway-url"):
        return
    skip = pytest.mark.skip(reason="needs --gateway-url (live gateway)")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


class GatewayClient:
    """Thin stdlib HTTP client for the gateway's observable surface."""

    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")

    def post(self, path: str, body: dict, timeout: float = 180.0):
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            self.url + path, data=data, headers={"Content-Type": "application/json"}
        )
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read()), time.monotonic() - t0

    def get_json(self, path: str, timeout: float = 10.0) -> dict:
        with urllib.request.urlopen(self.url + path, timeout=timeout) as resp:
            return json.loads(resp.read())

    def get_text(self, path: str, timeout: float = 10.0) -> str:
        with urllib.request.urlopen(self.url + path, timeout=timeout) as resp:
            return resp.read().decode("utf-8", "replace")

    def health(self) -> dict:
        return self.get_json("/health")

    @staticmethod
    def used_of(health: dict) -> int:
        return max((g.get("used_mb", 0) for g in health.get("gpus", [])), default=0)

    def used_mb(self) -> int:
        return self.used_of(self.health())

    def loaded(self) -> set:
        return set(self.health().get("loaded_models", []))

    def counter(self, name: str) -> int:
        for line in self.get_text("/metrics").splitlines():
            if line.startswith(name + " "):
                try:
                    return int(float(line.split()[1]))
                except (ValueError, IndexError):
                    return 0
        return 0


@pytest.fixture(scope="session")
def gw(request) -> GatewayClient:
    url = request.config.getoption("--gateway-url")
    if not url:
        pytest.skip("no --gateway-url; integration checks require a live gateway")
    client = GatewayClient(url)
    try:
        h = client.health()
    except (urllib.error.URLError, OSError) as exc:
        pytest.skip(f"gateway unreachable at {url}: {exc}")
    if h.get("status") != "ok":
        pytest.skip(f"gateway not healthy at {url}: {h}")
    return client
