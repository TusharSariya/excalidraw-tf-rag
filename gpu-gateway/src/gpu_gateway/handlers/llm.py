"""LLM deployment — HTTP proxy to Ollama.

Holds no GPU directly (num_gpus=0 in serve_config.yaml); Ollama manages its own VRAM.
OLLAMA_KEEP_ALIVE=0 must be set in Ollama's systemd unit so weights unload immediately
after each request, freeing VRAM before the next embed/rerank request acquires the lock.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request

from ray import serve

log = logging.getLogger("gpu_gateway.handlers.llm")

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


@serve.deployment
class LLMDeployment:
    def __init__(self) -> None:
        self._ollama_url = DEFAULT_OLLAMA_URL

    def reconfigure(self, config: dict) -> None:
        self._ollama_url = config.get("ollama_url", DEFAULT_OLLAMA_URL).rstrip("/")

    async def __call__(self, payload: dict) -> dict:
        """Forward an OpenAI-compatible /v1/chat/completions payload to Ollama's native API.

        Returns an OpenAI-compatible response including usage.prompt_tokens and
        usage.completion_tokens so callers like graphify can track token costs.
        """
        messages = payload.get("messages", [])
        model = payload.get("model", "gemma4:e4b")
        temperature = payload.get("temperature", 0.2)
        # Accept both OpenAI SDK key names
        max_tokens = payload.get("max_tokens") or payload.get("max_completion_tokens") or 512

        # Use Ollama native /api/chat with think:false (OpenAI compat ignores think flag)
        ollama_payload = {
            "model": model,
            "messages": messages,
            "think": False,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        url = f"{self._ollama_url}/api/chat"
        body = json.dumps(ollama_payload).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama unreachable at {self._ollama_url}: {exc}") from exc

        content = (data.get("message") or {}).get("content", "")
        # Ollama native API returns prompt_eval_count (input) and eval_count (output)
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        stop_reason = data.get("done_reason", "stop")
        finish_reason = "length" if stop_reason == "length" else "stop"
        # Return a complete OpenAI-compatible ChatCompletion shape.
        # id/created/object are required by the OpenAI Python SDK Pydantic model — callers
        # like graphify use the SDK client directly and will fail validation without them.
        return {
            "id": f"chatcmpl-gateway-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "model": model,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
