"""FastMCP server for the GPU Gateway — runs on Mac, stdio transport.

Wraps /v1/rerank and /health from the desktop gateway. Claude Code spawns this
via stdio; it calls the desktop over Tailscale (RAG_GPU_GATEWAY_URL).

Install on Mac:
  cd tools/gpu-gateway && uv sync --extra mcp

Configure in .claude/mcp.json:
  {
    "mcpServers": {
      "gpu-gateway": {
        "command": "uv",
        "args": ["run", "--project", "tools/gpu-gateway", "gpu-gateway-mcp"],
        "env": {"RAG_GPU_GATEWAY_URL": "http://desktop.tail12345.ts.net:8765"}
      }
    }
  }
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


def _gateway_url() -> str:
    url = os.environ.get("RAG_GPU_GATEWAY_URL", "").rstrip("/")
    if not url:
        raise RuntimeError("RAG_GPU_GATEWAY_URL is not set")
    return url


def _api_key() -> str:
    return os.environ.get("RAG_GPU_GATEWAY_KEY", "")


def _post(path: str, payload: dict) -> dict:
    url = f"{_gateway_url()}{path}"
    key = _api_key()
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json",
                                  **({"X-API-Key": key} if key else {})}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gateway {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Gateway unreachable at {url}: {exc}") from exc


def _get(path: str) -> dict:
    url = f"{_gateway_url()}{path}"
    key = _api_key()
    req = urllib.request.Request(url, headers={**({"X-API-Key": key} if key else {})})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gateway {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Gateway unreachable at {url}: {exc}") from exc


def main() -> None:
    from fastmcp import FastMCP

    mcp = FastMCP("gpu-gateway")

    @mcp.tool()
    def rerank(query: str, passages: list[str], model: str = "bge-reranker-v2-m3") -> list[dict]:
        """Rerank passages by cross-encoder relevance to query using the desktop GPU.

        Returns list of {index, score} sorted by the original passage order.
        Caller should sort by score descending to get the most relevant first.
        """
        result = _post("/v1/rerank", {"model": model, "query": query, "passages": passages})
        return result.get("results", [])

    @mcp.tool()
    def health() -> dict:
        """Check GPU gateway health: VRAM usage, loaded models, cache size."""
        return _get("/health")

    mcp.run()


if __name__ == "__main__":
    main()
