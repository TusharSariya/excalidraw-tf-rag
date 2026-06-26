"""HG1 — backend-match preflight assert.

Quantized local vectors are NOT portable across MLX (Apple Silicon) and
bitsandbytes (CUDA): a profile records backend "local" / quant "4bit" for both, so
the model/dims/quant check is blind to the split. Benchmarking a `cuda-*` index on a
Mac silently embeds MLX-quant queries against bnb-quant docs — the defect that
corrupted Stage A. These tests pin the fail-closed behavior.
"""

import warnings

import pytest

from rag_common.config import EmbedConfig, resolved_embed_backend
from graph_layout_rag.query.retrieve import _assert_backend_match


class _Paths:
    profile = "test-rung"


def _qwen_cfg() -> EmbedConfig:
    return EmbedConfig(
        backend="local",
        model="Qwen/Qwen3-Embedding-0.6B",
        dimensions=1024,
        profile="cuda-qwen0.6b-1024",
        quant="4bit",
    )


def test_matched_backend_passes():
    cfg = _qwen_cfg()
    host = resolved_embed_backend(cfg)
    # Recorded tag equals what this host resolves -> no raise.
    _assert_backend_match({"resolved_embed_backend": host}, cfg, _Paths())


def test_cross_quant_mismatch_raises():
    cfg = _qwen_cfg()
    host = resolved_embed_backend(cfg)
    other = "cuda-bnb-4bit" if host == "mlx-q4" else "mlx-q4"
    with pytest.raises(RuntimeError, match="HG1 backend mismatch"):
        _assert_backend_match({"resolved_embed_backend": other}, cfg, _Paths())


def test_legacy_index_warns_not_raises():
    cfg = _qwen_cfg()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _assert_backend_match({}, cfg, _Paths())  # no recorded tag
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_gemini_is_portable():
    cfg = EmbedConfig(
        backend="gemini",
        model="gemini-embedding-2-preview",
        dimensions=3072,
        profile="gemini-2-structure-v1",
        quant=None,
    )
    assert resolved_embed_backend(cfg) == "gemini"
    # Gemini (API, deterministic) is portable across hosts -> matches everywhere.
    _assert_backend_match({"resolved_embed_backend": "gemini"}, cfg, _Paths())


def test_resolved_tag_is_host_specific_for_qwen_4bit():
    # The same config must NOT resolve to a generic "local"; it must pick a concrete
    # quant stack so cross-host scoring can be caught.
    assert resolved_embed_backend(_qwen_cfg()) in ("mlx-q4", "cuda-bnb-4bit")
