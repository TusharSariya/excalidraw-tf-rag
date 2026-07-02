"""Pluggable text-generation backends for RAG query transforms and agents.

Supports local Ollama (default) via the OpenAI-compatible
``/v1/chat/completions`` API, plus opencode and GPU-gateway backends.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any

log = logging.getLogger("rag_common.local_llm")

DEFAULT_BACKEND = "ollama"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "gemma4:e4b"
# opencode CLI backend: shells out to `opencode run -m <model> --format json
# <prompt>` and collects the assistant text events. Lets the eval judge/synth
# run on opencode-routed models (e.g. opencode-go/deepseek-v4-flash) with no
# Vertex/GCP billing. The agent harness injects a system prompt (~20k tokens),
# so this is best for short grading/generation prompts, not bulk transforms.
DEFAULT_OPENCODE_MODEL = "opencode-go/deepseek-v4-flash"
DEFAULT_OPENCODE_BIN = "opencode"


def llm_backend() -> str:
    raw = os.getenv("RAG_LLM_BACKEND", DEFAULT_BACKEND).strip().lower()
    if raw in ("ollama", "opencode", "gateway"):
        return raw
    return DEFAULT_BACKEND


def ollama_host() -> str:
    return (os.getenv("RAG_OLLAMA_HOST", DEFAULT_OLLAMA_HOST)).strip().rstrip("/")


def ollama_model() -> str:
    return (os.getenv("RAG_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)).strip() or DEFAULT_OLLAMA_MODEL


def opencode_bin() -> str:
    return (os.getenv("RAG_OPENCODE_BIN", DEFAULT_OPENCODE_BIN)).strip() or DEFAULT_OPENCODE_BIN


def opencode_model() -> str:
    return (
        os.getenv("RAG_OPENCODE_MODEL", DEFAULT_OPENCODE_MODEL)
    ).strip() or DEFAULT_OPENCODE_MODEL


def active_model() -> str:
    backend = llm_backend()
    if backend == "opencode":
        return opencode_model()
    return ollama_model()


def model_slug(model: str | None = None) -> str:
    name = (model or active_model()).strip()
    slug = re.sub(r"[^\w.-]+", "_", name.lower())
    return slug.strip("_") or "default"


def transform_cache_filename(model: str | None = None) -> str:
    return f"transform_cache_{llm_backend()}_{model_slug(model)}.json"


def llm_metadata() -> dict[str, str]:
    return {"llm_backend": llm_backend(), "llm_model": active_model()}


def generate_text(
    prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    backend = llm_backend()
    if backend == "gateway":
        resolved = model or ollama_model()
        return _generate_gateway(
            prompt, model=resolved, temperature=temperature, max_tokens=max_tokens
        )
    if backend == "ollama":
        resolved = model or ollama_model()
        return _generate_ollama(
            prompt,
            model=resolved,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if backend == "opencode":
        resolved = model or opencode_model()
        return _generate_opencode(prompt, model=resolved)
    resolved = model or ollama_model()
    return _generate_ollama(prompt, model=resolved, temperature=temperature, max_tokens=max_tokens)


def _generate_gateway(
    prompt: str,
    *,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Generate via the GPU gateway /v1/chat/completions (proxied to Ollama on desktop)."""
    gateway_url = os.getenv("RAG_GPU_GATEWAY_URL", "").rstrip("/")
    if not gateway_url:
        raise RuntimeError("RAG_LLM_BACKEND=gateway but RAG_GPU_GATEWAY_URL is not set")
    url = f"{gateway_url}/v1/chat/completions"
    key = os.getenv("RAG_GPU_GATEWAY_KEY", "")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            **({"X-API-Key": key} if key else {}),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GPU gateway HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"GPU gateway unreachable at {gateway_url}: {exc}") from exc
    text = _extract_chat_text(data)
    if not text:
        raise RuntimeError(f"Empty GPU gateway response from {model}")
    return text


def _generate_opencode(prompt: str, *, model: str, timeout: float = 180.0) -> str:
    """Generate via the ``opencode`` CLI in headless JSON mode.

    Runs ``opencode run -m <model> --format json <prompt>`` and concatenates the
    text of every ``type == "text"`` event (``part.text``). The CLI streams
    JSON-lines on stdout; non-JSON / non-text events (step_start/step_finish,
    reasoning, tool calls) are ignored. ``temperature``/``max_tokens`` are not
    exposed by the CLI, so callers needing determinism should rely on the
    grade-parsing being robust to minor variation.
    """
    import subprocess

    args = [opencode_bin(), "run", "-m", model, "--format", "json", prompt]
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise RuntimeError(f"opencode CLI not found ({opencode_bin()})") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"opencode timed out after {timeout:.0f}s for {model}") from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"opencode exit {proc.returncode} for {model}: "
            f"{(proc.stderr or proc.stdout or '')[:300]}"
        )
    parts: list[str] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "text":
            text = (event.get("part") or {}).get("text")
            if text:
                parts.append(text)
    out = "".join(parts).strip()
    if not out:
        raise RuntimeError(f"Empty opencode response from {model}")
    return out


def _generate_ollama(
    prompt: str,
    *,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    # Use native /api/chat with think:false to disable Qwen3 thinking mode.
    # The OpenAI-compat endpoint doesn't support think:false, leaving content empty.
    url = f"{ollama_host()}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "think": False,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama unavailable at {ollama_host()}: {exc}") from exc

    text = (data.get("message") or {}).get("content", "").strip()
    if not text:
        raise RuntimeError(f"Empty Ollama response from {model}")
    return text


def _extract_chat_text(data: dict[str, Any]) -> str:
    import re as _re

    choices = data.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        # Qwen3 thinking mode wraps output in <think>…</think>; strip it and
        # use what follows. Also fall back to reasoning_content if content is empty.
        stripped = _re.sub(r"<think>.*?</think>", "", content, flags=_re.DOTALL).strip()
        if stripped:
            return stripped
        # Ollama uses "reasoning" (not "reasoning_content") for the thinking field
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        if reasoning.strip():
            return reasoning.strip()
    message = data.get("message") or {}
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    return ""
