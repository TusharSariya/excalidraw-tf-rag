from graph_layout_rag.eval.encode_device import fastembed_kwargs, resolve_encode_device
from graph_layout_rag.eval.experimental_index import (
    DEFAULT_MODELS,
    _clean_payload,
    _sparse_backend,
)
from graph_layout_rag.eval.splade_v3_encoder import PYTORCH_SPARSE_MODELS, is_pytorch_sparse_model


def test_experimental_models_are_pinned():
    assert DEFAULT_MODELS["splade"] == "prithivida/Splade_PP_en_v1"
    assert DEFAULT_MODELS["colbert"] == "answerdotai/answerai-colbert-small-v1"


def test_clean_payload_converts_nan_to_none():
    payload = _clean_payload({"year": float("nan"), "title": "x"})
    assert payload == {"year": None, "title": "x"}


def test_pytorch_sparse_model_detection():
    assert is_pytorch_sparse_model("naver/splade-v3")
    assert is_pytorch_sparse_model("naver/splade-v3-distilbert")
    assert not is_pytorch_sparse_model("prithivida/Splade_PP_en_v1")
    assert _sparse_backend("naver/splade-v3") == "pytorch_sparse"
    assert _sparse_backend("prithivida/Splade_PP_en_v1") == "fastembed"


def test_local_splade_checkpoint_routing(tmp_path):
    # A hub name that is not a known sparse model and not a local dir -> False.
    assert not is_pytorch_sparse_model("sentence-transformers/all-MiniLM-L6-v2")

    # A local dir with modules.json under a 'splade' path -> routed as pytorch sparse.
    ckpt = tmp_path / "checkpoints" / "splade-gd-v1"
    ckpt.mkdir(parents=True)
    (ckpt / "modules.json").write_text("[]")
    assert is_pytorch_sparse_model(str(ckpt))
    assert _sparse_backend(str(ckpt)) == "pytorch_sparse"

    # A local dir WITHOUT modules.json -> not misrouted.
    bare = tmp_path / "splade-empty"
    bare.mkdir()
    assert not is_pytorch_sparse_model(str(bare))

    # A modules.json dir whose path has no splade/training marker -> conservative False.
    other = tmp_path / "some" / "random-model"
    other.mkdir(parents=True)
    (other / "modules.json").write_text("[]")
    assert not is_pytorch_sparse_model(str(other))


def test_resolve_encode_device_defaults_cpu_without_cuda(monkeypatch):
    monkeypatch.delenv("GRAPH_RAG_FASTEMBED_CUDA", raising=False)
    monkeypatch.setenv("GRAPH_RAG_ENCODE_DEVICE", "cpu")
    assert resolve_encode_device() == "cpu"
    assert fastembed_kwargs("cpu") == {}


def test_pytorch_sparse_models_include_v3_variants():
    assert "naver/splade-v3" in PYTORCH_SPARSE_MODELS
    assert "naver/splade-v3-distilbert" in PYTORCH_SPARSE_MODELS
