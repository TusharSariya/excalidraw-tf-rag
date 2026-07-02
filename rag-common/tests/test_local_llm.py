import json
from unittest.mock import patch

import pytest
from rag_common import local_llm


def test_llm_backend_defaults_to_ollama():
    with patch.dict("os.environ", {}, clear=True):
        assert local_llm.llm_backend() == "ollama"


def test_llm_backend_ollama():
    with patch.dict("os.environ", {"RAG_LLM_BACKEND": "ollama"}, clear=True):
        assert local_llm.llm_backend() == "ollama"


def test_model_slug_sanitizes():
    assert local_llm.model_slug("qwen2.5:7b-instruct-q4_K_M") == "qwen2.5_7b-instruct-q4_k_m"


def test_transform_cache_filename_includes_backend_and_model():
    with patch.dict(
        "os.environ",
        {
            "RAG_LLM_BACKEND": "ollama",
            "RAG_OLLAMA_MODEL": "qwen2.5:7b",
        },
        clear=True,
    ):
        assert local_llm.transform_cache_filename() == "transform_cache_ollama_qwen2.5_7b.json"


def test_generate_text_ollama_parses_openai_response():
    payload = {
        "choices": [{"message": {"content": "  hypothetical passage  "}}],
    }

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(payload).encode("utf-8")

    with patch.dict(
        "os.environ",
        {"RAG_LLM_BACKEND": "ollama", "RAG_OLLAMA_MODEL": "qwen2.5:7b"},
        clear=True,
    ):
        with patch("urllib.request.urlopen", return_value=_FakeResponse()):
            text = local_llm.generate_text("write a passage")
    assert text == "hypothetical passage"


def test_generate_text_ollama_empty_raises():
    payload = {"choices": [{"message": {"content": ""}}]}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(payload).encode("utf-8")

    with patch.dict("os.environ", {"RAG_LLM_BACKEND": "ollama"}, clear=True):
        with patch("urllib.request.urlopen", return_value=_FakeResponse()):
            with pytest.raises(RuntimeError, match="Empty Ollama"):
                local_llm.generate_text("prompt")
