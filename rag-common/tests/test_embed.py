from unittest.mock import MagicMock, patch

from rag_common.config import local_config
from rag_common.embed import embed_texts, resolve_embed_config


@patch.dict("os.environ", {}, clear=True)
def test_auto_no_key_uses_local():
    cfg = resolve_embed_config()
    assert cfg.backend == "local"
    assert cfg.dimensions == 1024


@patch.dict("os.environ", {"RAG_EMBED_BACKEND": "local"}, clear=True)
def test_explicit_local():
    cfg = resolve_embed_config()
    assert cfg.backend == "local"


@patch.dict("os.environ", {"RAG_EMBED_BACKEND": "local"}, clear=True)
@patch("rag_common.local_embed.SentenceTransformer")
def test_embed_local(mock_st_cls):
    mock_model = MagicMock()
    mock_st_cls.return_value = mock_model
    import numpy as np

    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

    cfg = local_config()
    vectors = embed_texts(["hello"], config=cfg, allow_fallback=False)
    assert len(vectors) == 1
    assert len(vectors[0]) == 3
