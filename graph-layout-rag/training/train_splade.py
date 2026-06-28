"""Contrastive LoRA fine-tune of a released SPLADE checkpoint (no teacher).

Objective (NON-distillation):
    SpladeLoss(model, loss=SparseMultipleNegativesRankingLoss(model),
               query_regularizer_weight=5e-4, document_regularizer_weight=1e-4)
i.e. in-batch + hard-negative contrastive ranking + FLOPS sparsity regularization.

Base: opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill — the
only released SPLADE that loads cleanly under the eval env's transformers/torch ABI
(validated in _api_smoke.py). LoRA on the DistilBERT doc-encoder; the query branch
is an inference-free SparseStaticEmbedding (no params).

Serving honesty: after training we ``merge_and_unload`` the LoRA adapter into a
standalone checkpoint and ASSERT served weights != base (else the eval harness
silently serves the base = the M18 NULL).

    uv run --no-sync python train_splade.py --epochs 2 --out data/training/checkpoints/splade-gd-v1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_splade")

TOOL_ROOT = Path(__file__).resolve().parent.parent
# Data dir is env-overridable so a separate run (e.g. GATE-1) can use its own
# namespace of artifacts without clobbering the default data/training/ files.
TRAIN_DATA_DIR = Path(os.getenv("GRAPH_RAG_TRAIN_DATA_DIR", str(TOOL_ROOT / "data" / "training")))
TRIPLES_PATH = TRAIN_DATA_DIR / "triples.jsonl"
DEFAULT_OUT = TRAIN_DATA_DIR / "checkpoints" / "splade-gd-v1"

BASE_MODEL = "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill"
# DistilBERT attention projections (validated LoRA targets).
LORA_TARGETS = ["q_lin", "k_lin", "v_lin", "out_lin"]


# --------------------------------------------------------------------------- #
# Module discovery + served!=base probe (shared with _api_smoke.py)
# --------------------------------------------------------------------------- #
def transformer_module(model):
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


def weight_signature(model):
    import torch

    tmod = transformer_module(model)
    sig = []
    for _, p in tmod.auto_model.named_parameters():
        if p.numel() == 0:
            continue
        sig.append(p.detach().float().flatten()[:64].cpu())
        if len(sig) >= 8:
            break
    return torch.cat(sig)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_triples(path: Path, limit: int = 0):
    from datasets import Dataset

    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        negs = r.get("negative_texts") or []
        if not r.get("query") or not r.get("positive_text") or not negs:
            continue
        rows.append(r)
        if limit and len(rows) >= limit:
            break

    # SparseMultipleNegativesRankingLoss consumes (anchor, positive, negative...)
    # columns. We expose a single hard negative per row (the hardest) so every row
    # has a uniform schema; the remaining in-batch examples supply extra negatives.
    data = {
        "anchor": [r["query"] for r in rows],
        "positive": [r["positive_text"] for r in rows],
        "negative": [r["negative_texts"][0] for r in rows],
    }
    return Dataset.from_dict(data)


def attach_lora(model, *, r: int, alpha: int, dropout: float):
    from peft import LoraConfig, get_peft_model

    tmod = transformer_module(model)
    hf = tmod.auto_model
    present = {n.split(".")[-1] for n, _ in hf.named_modules()}
    targets = [t for t in LORA_TARGETS if t in present]
    if not targets:
        raise RuntimeError(f"none of {LORA_TARGETS} present in base encoder")
    peft_model = get_peft_model(
        hf, LoraConfig(r=r, lora_alpha=alpha, target_modules=targets, lora_dropout=dropout)
    )
    tmod.auto_model = peft_model
    n_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    log.info("LoRA attached targets=%s trainable=%d", targets, n_trainable)
    return peft_model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", default=str(TRIPLES_PATH))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="override epochs (e.g. 1 for the tiny E2E smoke)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--query-reg", type=float, default=5e-4)
    ap.add_argument("--doc-reg", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--eval-frac", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from sentence_transformers import SparseEncoder
    from sentence_transformers.sparse_encoder import (SparseEncoderTrainer,
                                                      SparseEncoderTrainingArguments)
    from sentence_transformers.sparse_encoder.losses import (
        SpladeLoss, SparseMultipleNegativesRankingLoss)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    log.info("loading base %s on %s", args.base, device)
    model = SparseEncoder(args.base, device=device)
    base_sig = weight_signature(model).clone()

    peft_model = attach_lora(model, r=args.lora_r, alpha=args.lora_alpha,
                             dropout=args.lora_dropout)

    ds = load_triples(Path(args.triples), limit=args.limit)
    log.info("loaded %d training rows", len(ds))
    eval_ds = None
    if args.eval_frac > 0 and len(ds) >= 20:
        split = ds.train_test_split(test_size=args.eval_frac, seed=42)
        ds, eval_ds = split["train"], split["test"]

    loss = SpladeLoss(
        model,
        loss=SparseMultipleNegativesRankingLoss(model),
        query_regularizer_weight=args.query_reg,
        document_regularizer_weight=args.doc_reg,
    )

    # The base is a Router (asymmetric query/doc SPLADE): the trainer must know
    # which dataset columns feed the inference-free query branch vs the trainable
    # document branch. anchor=query side; positive/negative=document side.
    router_mapping = {"anchor": "query", "positive": "document", "negative": "document"}

    targs = SparseEncoderTrainingArguments(
        output_dir=str(out / "_trainer"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        router_mapping=router_mapping,
        logging_steps=20,
        save_strategy="no",
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=50 if eval_ds is not None else None,
        load_best_model_at_end=False,
        fp16=device == "cuda",
        report_to=[],
    )

    trainer = SparseEncoderTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        eval_dataset=eval_ds,
        loss=loss,
    )
    log.info("training...")
    trainer.train()

    # Merge LoRA -> standalone checkpoint. Use the held peft_model reference: the
    # MLMTransformer's .auto_model attribute does not reliably round-trip the
    # PeftModel wrapper (it reads back as the bare HF model).
    tmod = transformer_module(model)
    merged = peft_model.merge_and_unload()
    tmod.auto_model = merged
    model.save_pretrained(str(out))
    log.info("saved merged checkpoint -> %s", out)

    # Serving honesty: reload through SparseEncoder + served!=base probe.
    reloaded = SparseEncoder(str(out), device=device)
    emb = reloaded.encode_document(["network simplex layered graph drawing"],
                                   convert_to_sparse_tensor=True)
    nnz = int(emb[0].coalesce().values().numel())
    merged_sig = weight_signature(reloaded)
    delta = float((merged_sig - base_sig).abs().max())
    assert nnz > 0, "merged checkpoint produced empty sparse encoding"
    assert delta > 1e-6, f"served weights == base (LoRA delta lost): delta={delta}"

    summary = {
        "base": args.base,
        "out": str(out),
        "train_rows": len(ds),
        "eval_rows": len(eval_ds) if eval_ds is not None else 0,
        "merged_nnz": nnz,
        "served_vs_base_max_abs_delta": delta,
        "served_not_base": True,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "targets": LORA_TARGETS},
        "reg": {"query": args.query_reg, "doc": args.doc_reg},
    }
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print("TRAIN_SUMMARY:", json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
