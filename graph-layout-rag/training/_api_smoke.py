"""CRITICAL pre-build API validation for the contrastive SPLADE training stack.

Confirms, on the real (desktop) training env, that every risky assumption holds
BEFORE any data is generated:

  1. sentence-transformers exposes SparseEncoder, SpladeLoss,
     SparseMultipleNegativesRankingLoss (the NON-distillation contrastive base loss).
  2. A released SPLADE checkpoint loads as a SparseEncoder and encodes non-empty.
     Prefers opensearch-...-doc-v3-gte (harness already routes opensearch-*),
     falls back to naver/splade-v3.
  3. LoRA/PEFT attaches to the SparseEncoder's transformer.
  4. After merge_and_unload + save, the merged dir loads back through
     SparseEncoder(<path>) AND its weights DIFFER from the base (served != base).

Any failure here = STOP-and-report. Prints a machine-readable RESULT: line.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch


def _imports() -> dict:
    out = {}
    import sentence_transformers as st

    out["st_version"] = st.__version__
    out["torch_version"] = torch.__version__
    out["cuda"] = torch.cuda.is_available()
    from sentence_transformers import SparseEncoder  # noqa: F401
    from sentence_transformers.sparse_encoder.losses import (  # noqa: F401
        SpladeLoss,
        SparseMultipleNegativesRankingLoss,
    )

    out["has_SparseEncoder"] = True
    out["has_SpladeLoss"] = True
    out["has_SparseMNRL"] = True
    import peft  # noqa: F401

    out["peft_version"] = peft.__version__
    return out


# Candidate SPLADE bases in preference order.
# v3-distill is the ONLY one that loads cleanly under the eval env's
# transformers 5.11 + torch<2.6 ABI (safetensors, plain BERT-MLM, no remote code,
# harness routes opensearch-*). The naver/.bin checkpoints are blocked by the
# transformers torch.load>=2.6 gate; the -gte variant device-side-asserts / IndexErrors
# under transformers 5.x custom code. See _base_probe.py results.
CANDIDATES = [
    "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill",
]


def _load_base():
    from sentence_transformers import SparseEncoder

    last_err = None
    for name in CANDIDATES:
        try:
            model = SparseEncoder(name, device="cuda" if torch.cuda.is_available() else "cpu")
            return name, model
        except Exception as exc:  # noqa: BLE001
            last_err = f"{name}: {type(exc).__name__}: {exc}"
            print(f"  base load FAILED {last_err}", flush=True)
    raise RuntimeError(f"no SPLADE base loaded; last={last_err}")


def _transformer_module(model):
    """Return the module that owns ``.auto_model`` (the HF MLM transformer).

    The opensearch-doc SPLADE is a ``Router`` with a query branch
    (``SparseStaticEmbedding`` — inference-free, no params to train) and a document
    branch whose first element is an ``MLMTransformer`` exposing ``.auto_model``
    (``DistilBertForMaskedLM``). Recurse to find it.
    """

    def _walk(mod):
        if hasattr(mod, "auto_model") and mod.auto_model is not None:
            return mod
        for sub in mod._modules.values():
            if sub is None:
                continue
            found = _walk(sub)
            if found is not None:
                return found
        return None

    tmod = _walk(model)
    if tmod is None:
        raise RuntimeError("no module with .auto_model found inside SparseEncoder")
    return tmod


def _weight_signature(model) -> torch.Tensor:
    tmod = _transformer_module(model)
    # Flatten a deterministic slice of the encoder weights to a CPU vector.
    sig = []
    for n, p in tmod.auto_model.named_parameters():
        if p.numel() == 0:
            continue
        sig.append(p.detach().float().flatten()[:64].cpu())
        if len(sig) >= 8:
            break
    return torch.cat(sig)


def main() -> int:
    result = {"ok": False}
    try:
        imp = _imports()
        result.update(imp)
        print("IMPORTS OK:", json.dumps(imp), flush=True)

        base_name, model = _load_base()
        result["base"] = base_name
        print(f"LOADED BASE: {base_name}", flush=True)

        # Encode a doc -> non-empty sparse vector.
        emb = model.encode_document(
            ["network simplex rank assignment in layered graph drawing"],
            convert_to_sparse_tensor=True,
        )
        nnz = int(emb[0].coalesce().values().numel())
        result["base_nnz"] = nnz
        assert nnz > 0, "base produced empty sparse encoding"
        print(f"BASE ENCODE nnz={nnz}", flush=True)

        base_sig = _weight_signature(model).clone()

        # Attach LoRA to the HF transformer.
        from peft import LoraConfig, get_peft_model

        tmod = _transformer_module(model)
        hf = tmod.auto_model
        # Target common attention/MLP projections present in BERT/ModernBERT/GTE.
        candidate_targets = [
            ["q_lin", "k_lin", "v_lin", "out_lin"],      # DistilBERT (opensearch v3-distill)
            ["query", "key", "value", "dense"],          # BERT-family
            ["Wqkv", "Wo", "Wi"],                         # ModernBERT
            ["q_proj", "k_proj", "v_proj", "o_proj"],    # GTE/llama-ish
        ]
        attached = None
        peft_model = None
        for targets in candidate_targets:
            present = {n.split(".")[-1] for n, _ in hf.named_modules()}
            if not any(t in present for t in targets):
                continue
            try:
                lconf = LoraConfig(r=8, lora_alpha=16, target_modules=targets, lora_dropout=0.0)
                peft_model = get_peft_model(hf, lconf)
                tmod.auto_model = peft_model
                attached = targets
                break
            except Exception as exc:  # noqa: BLE001
                print(f"  LoRA targets {targets} failed: {exc}", flush=True)
        assert attached is not None and peft_model is not None, (
            "LoRA could not attach to any known target module set"
        )
        result["lora_targets"] = attached
        n_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        result["lora_trainable_params"] = n_trainable
        assert n_trainable > 0
        print(f"LORA ATTACHED targets={attached} trainable={n_trainable}", flush=True)

        # Perturb the LoRA params so merge produces a real delta (simulate training).
        with torch.no_grad():
            for n, p in peft_model.named_parameters():
                if "lora_B" in n and p.requires_grad:
                    p.add_(torch.randn_like(p) * 0.02)

        # Merge and save. Hold the peft model in a local var (the MLMTransformer's
        # .auto_model attribute does not reliably round-trip the PeftModel wrapper).
        merged_hf = peft_model.merge_and_unload()
        tmod.auto_model = merged_hf
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "merged"
            model.save_pretrained(str(out))
            print(f"SAVED MERGED -> {out}", flush=True)

            from sentence_transformers import SparseEncoder

            reloaded = SparseEncoder(str(out), device="cuda" if torch.cuda.is_available() else "cpu")
            remb = reloaded.encode_document(
                ["network simplex rank assignment in layered graph drawing"],
                convert_to_sparse_tensor=True,
            )
            rnnz = int(remb[0].coalesce().values().numel())
            result["merged_nnz"] = rnnz
            assert rnnz > 0, "merged reload produced empty sparse encoding"

            merged_sig = _weight_signature(reloaded)
            # served != base probe
            max_abs_delta = float((merged_sig - base_sig).abs().max())
            result["served_vs_base_max_abs_delta"] = max_abs_delta
            assert max_abs_delta > 1e-6, "served weights == base (LoRA delta lost on merge/reload)"
            print(
                f"RELOAD MERGED nnz={rnnz} served!=base delta={max_abs_delta:.6g}",
                flush=True,
            )

        result["ok"] = True
    except Exception as exc:  # noqa: BLE001
        import traceback

        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        print("FAILED:", result["error"], flush=True)
        print(result["traceback"], flush=True)
    print("RESULT:", json.dumps(result), flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
