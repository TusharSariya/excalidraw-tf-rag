"""Dense embedding deployments (Qwen3-4B and Qwen3-0.6B).

user_config is delivered via reconfigure() in Ray Serve 2.x — NOT __init__.
Models are loaded LAZILY on first __call__ and EVICTED by the Scheduler before it
admits a model that doesn't fit (and always before LLM requests). Eviction/park
mechanics live in the shared _EvictableModel mixin.

`__call__` is token-aware: incoming texts are split into sub-batches capped by BOTH
the handler's item-count batch_size AND an estimated token budget (see
`gpu_gateway.token_batch`), and each sub-batch's `.encode()` call is wrapped with
CUDA OOM recovery (empty_cache + halve + retry) so one batch of unusually long
chunks degrades gracefully instead of OOM'ing the whole replica — see
`_encode_sub_batch`. Defaults reproduce the pre-existing fixed-batch_size behavior
byte-for-byte when inputs are short enough that the token budget never binds.
"""

from __future__ import annotations

import logging

from ray import serve

from gpu_gateway.handlers.evictable import _EvictableModel
from gpu_gateway.token_batch import (
    EmbedOOMError,
    embed_token_budget,
    estimate_tokens,
    is_cuda_oom,
    split_token_aware,
)

log = logging.getLogger("gpu_gateway.handlers.dense_embed")

__all__ = ["DenseEmbed4B", "DenseEmbed06B", "EmbedOOMError"]


class _DenseEmbedBase(_EvictableModel):
    """Shared implementation; subclassed for 4B and 0.6B deployments."""

    # Per-item batch_size cap; subclasses override to match their VRAM headroom.
    _batch_size: int = 32

    def __init__(self) -> None:
        self._model = None
        self._model_path: str = ""

    def reconfigure(self, config: dict) -> None:
        # Store the model path; actual loading is deferred to the first __call__.
        self._model_path = config["model"]
        self._model = None  # reset so __call__ reloads when config changes
        self._parked = False
        self._configure_evict(config)

    def _load_from_disk(self) -> None:
        from sentence_transformers import SentenceTransformer

        log.info("lazy-loading dense embed model %s", self._model_path)
        self._model = SentenceTransformer(self._model_path, device="cuda")
        # Cap input sequence length — long code chunks produce O(seq²) attention
        # tensors that OOM an 8 GB card even at batch_size=32.
        self._model.max_seq_length = 2048
        log.info("loaded %s", self._model_path)

    def _tokenizer(self):
        """The underlying HF tokenizer if the model is loaded, else None (falls
        back to the chars/4 heuristic in `token_batch.estimate_tokens`)."""
        return getattr(self._model, "tokenizer", None)

    def _empty_cuda_cache(self) -> None:
        import gc

        import torch

        gc.collect()  # break SentenceTransformer's module cycles before empty_cache
        torch.cuda.empty_cache()

    def _encode_sub_batch(self, sub_batch: list[str], batch_size: int) -> list[list[float]]:
        """Encode one sub-batch. On CUDA OOM: empty the allocator cache and retry
        with the sub-batch halved (and internal batch_size halved to match). A
        single item that still OOMs raises EmbedOOMError rather than propagating
        the raw CUDA exception up through Ray Serve."""
        try:
            vecs = self._model.encode(
                sub_batch,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=max(1, min(batch_size, len(sub_batch))),
            )
            return vecs.tolist()
        except Exception as exc:  # noqa: BLE001 — narrowed to CUDA OOM below; re-raised otherwise
            if not is_cuda_oom(exc):
                raise
            log.warning(
                "%s: CUDA OOM on sub-batch of %d (batch_size=%d); clearing cache and retrying",
                self._model_path,
                len(sub_batch),
                batch_size,
            )
            self._empty_cuda_cache()
            if len(sub_batch) == 1:
                raise EmbedOOMError(
                    f"{self._model_path}: single item OOM'd "
                    f"(~{estimate_tokens(sub_batch[0], self._tokenizer())} est. tokens); "
                    "cannot shrink batch further"
                ) from exc
            mid = max(1, len(sub_batch) // 2)
            half_batch_size = max(1, batch_size // 2)
            left = self._encode_sub_batch(sub_batch[:mid], half_batch_size)
            right = self._encode_sub_batch(sub_batch[mid:], half_batch_size)
            return left + right

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        self._ensure_loaded()
        # Sub-batch by BOTH item count (self._batch_size, unchanged from before) AND
        # an estimated token budget, so a handful of long docling chunks can't ride
        # along with the full item-count allowance and blow up padding-amplified
        # attention memory (batch_size x longest_seq^2). Short inputs that never hit
        # the token budget split exactly as before (one sub-batch per _batch_size
        # items), so default output is unchanged.
        sub_batches = split_token_aware(
            texts,
            max_items=self._batch_size,
            max_tokens=embed_token_budget(),
            tokenizer=self._tokenizer(),
        )
        results: list[list[float]] = []
        for sub_batch in sub_batches:
            results.extend(self._encode_sub_batch(sub_batch, self._batch_size))
        return results


@serve.deployment
class DenseEmbed4B(_DenseEmbedBase):
    def _load_from_disk(self) -> None:
        from sentence_transformers import SentenceTransformer
        from transformers import BitsAndBytesConfig

        # Qwen3-4B is ~8 GB in bf16 — OOMs on an 8 GB card.
        # BitsAndBytesConfig(load_in_8bit=True) reduces to ~4 GB.
        bnb = BitsAndBytesConfig(load_in_8bit=True)
        log.info("lazy-loading dense embed model %s (int8, 1024-dim MRL)", self._model_path)
        self._model = SentenceTransformer(
            self._model_path,
            device="cuda",
            truncate_dim=1024,
            model_kwargs={"quantization_config": bnb},
        )
        # 4B at int8 ≈ 5.7 GB; only ~2 GB left for activations.
        # seq=256 keeps FFN activations to <100 MB per layer.
        self._model.max_seq_length = 256
        log.info("loaded %s", self._model_path)

    # batch_size=8, seq=256: FFN activation = 8 × 256 × 8192 × 2B ≈ 32 MB/layer —
    # well within the ~1.5 GB headroom left after loading 5.7 GB of int8 weights.
    _batch_size = 8

    def _validate_after_unpark(self) -> bool:
        """bitsandbytes int8 layers don't reliably survive .to('cpu')→.to('cuda').
        Run a 1-token self-test: correct shape + finite values, else fall back."""
        import math

        try:
            v = self._model.encode(
                ["ok"], normalize_embeddings=True, show_progress_bar=False, batch_size=1
            )
            vec = list(v[0])
            if len(vec) != 1024:
                return False
            return all(math.isfinite(float(x)) for x in vec[:16])
        except Exception as exc:  # noqa: BLE001
            log.warning("4B unpark self-test raised: %s", exc)
            return False


@serve.deployment
class DenseEmbed06B(_DenseEmbedBase):
    # batch_size=4: 0.6B FFN intermediate is large; batch_size=32 OOMs
    # (28 layers × 32 × 512 × 4096 × 2B ≈ 7 GB of activations alone).
    _batch_size = 4

    def _load_from_disk(self) -> None:
        from sentence_transformers import SentenceTransformer

        log.info("lazy-loading dense embed model %s", self._model_path)
        self._model = SentenceTransformer(self._model_path, device="cuda")
        # 512 tokens: attention scales as seq², and 2048 OOMs an 8 GB card.
        self._model.max_seq_length = 512
        log.info("loaded %s", self._model_path)
