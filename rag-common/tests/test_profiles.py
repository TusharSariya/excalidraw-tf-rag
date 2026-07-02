from pathlib import Path
from unittest.mock import patch

import pytest
from rag_common.config import EmbedConfig
from rag_common.profiles import get_profile, load_profiles, resolve_embed_profile_config


def test_load_builtin_profiles():
    profiles = load_profiles()
    assert "cuda-qwen4b-1024" in profiles
    assert profiles["cuda-qwen4b-1024"].model == "Qwen/Qwen3-Embedding-4B"
    assert profiles["cuda-qwen4b-1024"].dimensions == 1024
    assert profiles["cuda-qwen4b-1024"].quant == "4bit"
    assert profiles["cuda-qwen4b-2560"].dimensions == 2560
    for name in [
        "cuda-qwen0.6b-section-v1",
        "cuda-qwen0.6b-contextual-v1",
        "cuda-qwen0.6b-small2big-v1",
    ]:
        assert profiles[name].model == "Qwen/Qwen3-Embedding-0.6B"
        assert profiles[name].dimensions == 1024
        assert profiles[name].quant == "4bit"
    assert profiles["cuda-qwen0.6b-512-v1"].model == "Qwen/Qwen3-Embedding-0.6B"
    assert profiles["cuda-qwen0.6b-512-v1"].dimensions == 512
    assert profiles["cuda-qwen0.6b-512-v1"].quant == "4bit"
    assert profiles["cuda-qwen0.6b-768-v1"].model == "Qwen/Qwen3-Embedding-0.6B"
    assert profiles["cuda-qwen0.6b-768-v1"].dimensions == 768
    assert profiles["cuda-qwen0.6b-768-v1"].quant == "4bit"


def test_merge_extra_profiles(tmp_path: Path):
    override = tmp_path / "embed_profiles.toml"
    override.write_text(
        """
[profiles.custom-local]
backend = "local"
model = "Qwen/Qwen3-Embedding-0.6B"
dimensions = 512
quant = "4bit"
""",
        encoding="utf-8",
    )
    profiles = load_profiles(extra_paths=[override])
    assert profiles["custom-local"].dimensions == 512


@patch.dict("os.environ", {}, clear=True)
def test_legacy_env_local_default():
    cfg = resolve_embed_profile_config()
    assert cfg.backend == "local"
    assert cfg.profile == "default"
    assert cfg.dimensions == 1024


@patch.dict("os.environ", {"RAG_EMBED_PROFILE": "cuda-qwen4b-1024"}, clear=True)
def test_profile_env_resolution():
    cfg = resolve_embed_profile_config()
    assert cfg.backend == "local"
    assert cfg.profile == "cuda-qwen4b-1024"
    assert cfg.model == "Qwen/Qwen3-Embedding-4B"
    assert cfg.dimensions == 1024


@patch.dict("os.environ", {"GRAPH_RAG_EMBED_PROFILE": "cuda-qwen4b-1024"}, clear=True)
def test_prefix_profile_env():
    cfg = resolve_embed_profile_config(prefix="GRAPH_RAG_")
    assert cfg.profile == "cuda-qwen4b-1024"
    assert cfg.quant == "4bit"


@patch.dict("os.environ", {}, clear=True)
def test_cli_profile_overrides_env():
    with patch.dict("os.environ", {"RAG_EMBED_PROFILE": "cuda-qwen4b-1024"}, clear=False):
        cfg = resolve_embed_profile_config(profile="qwen3-code")
    assert cfg.profile == "qwen3-code"
    assert cfg.backend == "local"


@patch.dict("os.environ", {"RAG_EMBED_BACKEND": "openai"}, clear=True)
def test_legacy_unknown_backend_rejected():
    with pytest.raises(RuntimeError, match="Unknown embed backend mode"):
        resolve_embed_profile_config()


@patch.dict(
    "os.environ",
    {"RAG_EMBED_BACKEND": "local", "RAG_LOCAL_EMBED_QUANT": "4bit"},
    clear=True,
)
def test_legacy_local_quant():
    cfg = resolve_embed_profile_config()
    assert cfg.backend == "local"
    assert cfg.quant == "4bit"


def test_get_profile_unknown():
    with pytest.raises(KeyError, match="Unknown embed profile"):
        get_profile("does-not-exist")


def test_profile_to_embed_config():
    spec = get_profile("qwen3-code")
    cfg = spec.to_embed_config()
    assert isinstance(cfg, EmbedConfig)
    assert cfg.profile == "qwen3-code"
    assert cfg.backend == "local"
