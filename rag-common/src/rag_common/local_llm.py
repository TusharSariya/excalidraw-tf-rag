"""Pluggable text-generation backends for RAG query transforms and agents.

Supports Vertex/Gemini (default, backward compatible) and local Ollama via the
OpenAI-compatible ``/v1/chat/completions`` API.
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

DEFAULT_BACKEND = "gemini"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "gemma4:e4b"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
# Apple-Silicon MLX path: an ``mlx_lm.server`` (OpenAI-compatible) serving a
# native MLX checkpoint. Used to run a local judge/transform on the Mac where
# the GGUF llama-server is too memory-heavy (a 12B GGUF RSS'd ~10GB and thrashed
# a 24GB host; the MLX 4-bit equivalent is ~7GB unified). Default port matches
# ``mlx_lm.server``'s default (8080).
DEFAULT_MLX_HOST = "http://127.0.0.1:8080"
DEFAULT_MLX_MODEL = "mlx-community/gemma-4-12B-it-qat-4bit"
# opencode CLI backend: shells out to `opencode run -m <model> --format json
# <prompt>` and collects the assistant text events. Lets the eval judge/synth
# run on opencode-routed models (e.g. opencode-go/deepseek-v4-flash) with no
# Vertex/GCP billing. The agent harness injects a system prompt (~20k tokens),
# so this is best for short grading/generation prompts, not bulk transforms.
DEFAULT_OPENCODE_MODEL = "opencode-go/deepseek-v4-flash"
DEFAULT_OPENCODE_BIN = "opencode"


def llm_backend() -> str:
    raw = os.getenv("RAG_LLM_BACKEND", DEFAULT_BACKEND).strip().lower()
    if raw in ("gemini", "ollama", "mlx", "opencode"):
        return raw
    return DEFAULT_BACKEND


def ollama_host() -> str:
    return (os.getenv("RAG_OLLAMA_HOST", DEFAULT_OLLAMA_HOST)).strip().rstrip("/")


def ollama_model() -> str:
    return (os.getenv("RAG_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)).strip() or DEFAULT_OLLAMA_MODEL


def mlx_host() -> str:
    return (os.getenv("RAG_MLX_HOST", DEFAULT_MLX_HOST)).strip().rstrip("/")


def mlx_model() -> str:
    return (os.getenv("RAG_MLX_MODEL", DEFAULT_MLX_MODEL)).strip() or DEFAULT_MLX_MODEL


def opencode_bin() -> str:
    return (os.getenv("RAG_OPENCODE_BIN", DEFAULT_OPENCODE_BIN)).strip() or DEFAULT_OPENCODE_BIN


def opencode_model() -> str:
    return (os.getenv("RAG_OPENCODE_MODEL", DEFAULT_OPENCODE_MODEL)).strip() or DEFAULT_OPENCODE_MODEL


def gemini_model() -> str:
    for key in (
        "GRAPH_RAG_AGENT_LLM_MODEL",
        "GRAPH_RAG_EVAL_LLM_MODEL",
        "GRAPH_RAG_CONTEXT_LLM_MODEL",
        "RAG_LIT_AGENT_LLM_MODEL",
        "RAG_LIT_EVAL_LLM_MODEL",
        "RAG_LIT_CONTEXT_LLM_MODEL",
        "RAG_LLM_MODEL",
    ):
        value = os.getenv(key, "").strip()
        if value:
            return value
    return DEFAULT_GEMINI_MODEL


def active_model() -> str:
    backend = llm_backend()
    if backend == "ollama":
        return ollama_model()
    if backend == "mlx":
        return mlx_model()
    if backend == "opencode":
        return opencode_model()
    return gemini_model()


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
    if backend == "ollama":
        resolved = model or ollama_model()
        return _generate_ollama(
            prompt,
            model=resolved,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if backend == "mlx":
        resolved = model or mlx_model()
        return _generate_mlx(
            prompt,
            model=resolved,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if backend == "opencode":
        resolved = model or opencode_model()
        return _generate_opencode(prompt, model=resolved)
    resolved = model or gemini_model()
    return _generate_gemini(prompt, model=resolved)


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
        proc = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout
        )
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


def _generate_mlx(
    prompt: str,
    *,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Generate via a local ``mlx_lm.server`` (OpenAI-compatible /v1/chat).

    mlx_lm.server serves whatever checkpoint it was launched with; the ``model``
    field is advisory. Reuses :func:`_extract_chat_text` (handles the OpenAI
    ``choices`` shape and strips any ``<think>`` block).
    """
    url = f"{mlx_host()}/v1/chat/completions"
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
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"mlx_lm.server HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"mlx_lm.server unavailable at {mlx_host()}: {exc}") from exc

    text = _extract_chat_text(data)
    if not text:
        raise RuntimeError(f"Empty mlx_lm.server response from {model}")
    return text


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


def _generate_gemini(prompt: str, *, model: str) -> str:
    try:
        from rag_common.gemini_embed import _client, llm_location
    except ImportError as exc:
        raise RuntimeError("Gemini client unavailable; install rag-common[gemini]") from exc

    client = _client(location=llm_location())
    response = client.models.generate_content(model=model, contents=prompt)
    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise RuntimeError(f"Empty LLM response from {model}")
    return text
