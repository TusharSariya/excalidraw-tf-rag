"""_EvictableModel — shared park/unpark/destroy mixin for GPU model handlers.

The dense embedders and the reranker share no base class, but they need identical
eviction mechanics: on evict, either destroy the model (default) or park it in CPU
RAM (`.to('cpu')`) so the next load is a fast `.to('cuda')` instead of a from-disk
reload. Writing this once keeps the fragile bitsandbytes device-move bits from
drifting across two files (review Issue 3 / Codex #15/#16).

A host class must provide:
  - `self._model`      — the live torch module (or None)
  - `self._model_path` — for logging
  - `_load_from_disk()` — construct `self._model` on cuda from scratch

The RAM cache is opt-in per deployment via `keep_on_cpu_after_evict` in user_config
(default off, matching the project's opt-in-toggle convention). `unload()` returns
True iff the model was parked (vs destroyed) so the Scheduler can track parked state.
"""

from __future__ import annotations

import logging

log = logging.getLogger("gpu_gateway.handlers.evictable")


class _EvictableModel:
    _keep_on_cpu: bool = False
    _parked: bool = False
    _park_failed: bool = False

    def _configure_evict(self, config: dict) -> None:
        self._keep_on_cpu = bool(config.get("keep_on_cpu_after_evict", False))

    # -- load path ------------------------------------------------------
    def _ensure_loaded(self) -> None:
        """Prefer unpark (fast) over a from-disk load; construct only if truly gone."""
        if self._model is not None and self._parked:
            self._unpark()
            return
        if self._model is None:
            self._load_from_disk()

    def _unpark(self) -> None:
        import gc

        import torch

        self._model.to("cuda")
        self._parked = False
        if self._validate_after_unpark():
            log.info("unparked %s from RAM", self._model_path)
            return
        # Device move corrupted the model (bnb int8 is the known offender): destroy,
        # reload from disk, and disable parking so future evictions take the safe path.
        log.warning(
            "%s: post-unpark self-test FAILED — reloading from disk, disabling RAM cache",
            self._model_path,
        )
        self._park_failed = True
        self._keep_on_cpu = False
        del self._model
        self._model = None
        gc.collect()  # break SentenceTransformer's module cycles before empty_cache
        torch.cuda.empty_cache()
        self._load_from_disk()

    def _validate_after_unpark(self) -> bool:
        """Base handlers (bf16 dense, cross-encoder) survive device moves cleanly.
        DenseEmbed4B overrides this with a real self-test (bnb int8 is fragile)."""
        return True

    # -- evict path -----------------------------------------------------
    async def unload(self) -> bool:
        import gc

        import torch

        if self._model is None:
            return False
        if self._keep_on_cpu:
            self._model.to("cpu")
            self._parked = True
            gc.collect()
            torch.cuda.empty_cache()
            log.info("parked %s in RAM (evict)", self._model_path)
            return True
        # A bare `del` is not enough: SentenceTransformer/transformers modules hold
        # circular references (module ↔ parent), so the weights aren't collected until
        # the cyclic GC runs — and empty_cache() only reclaims memory with zero live
        # refs. Without gc.collect() the CUDA allocation lingers (bnb int8 especially),
        # leaving a ghost the Scheduler can't see and eventually deadlocking the
        # measured-VRAM gate. See tests/test_integration.py::test_v1_no_vram_breach.
        del self._model
        self._model = None
        self._parked = False
        gc.collect()
        torch.cuda.empty_cache()
        log.info("unloaded %s (evict)", self._model_path)
        return False

    def ram_cache_status(self) -> str:
        if self._park_failed:
            return "disabled-after-failure"
        return "enabled" if self._keep_on_cpu else "off"
