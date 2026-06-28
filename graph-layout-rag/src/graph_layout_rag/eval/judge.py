"""LLM relevance judging of the multi-system pool (UMBRELA-style).

Grades every (query, pooled-doc) pair on the UMBRELA 0-3 scale with a single
deterministic LLM call, **blind to which retriever surfaced the doc** (so the
judge can't favor a retrieval style). Produces a graded qrels file that
`qrels.apply_qrels_overlay` folds into the gold set.

Reference: Upadhyay/Thomas et al., *UMBRELA* (arXiv:2406.06519) — graded 0-3
prompting reaches Kendall τ > 0.87 vs human assessors on TREC DL and gives high
system-ranking correlation.

Concurrency + on-disk checkpointing mirror `ingest/contextual.py` so a full pool
(a few thousand judgments) is feasible and re-runs are cheap.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from graph_layout_rag.paths import DATA_DIR

log = logging.getLogger("graph_layout_rag.eval.judge")

JUDGE_MODEL_ENV = "GRAPH_RAG_JUDGE_LLM_MODEL"
DEFAULT_JUDGE_MODEL = "gemini-3.1-pro-preview"
# Cache path is env-overridable so concurrent judge runs on different backends
# (e.g. Mac MLX ‖ desktop CUDA, validated in parallel) write SEPARATE files and
# never lost-update each other — `_save_cache` rewrites the whole file, so two
# writers sharing one path would clobber. Grades are model-keyed regardless.
CACHE_PATH = Path(
    os.getenv("GRAPH_RAG_JUDGE_CACHE", str(DATA_DIR / "eval" / "judge_cache.json"))
)
MAX_ABSTRACT_CHARS = 1500
MAX_EXCERPT_CHARS = 800

# Retry/backoff for transient LLM-judge failures. Without this, a 429 quota
# burst was caught and frozen into the cache as a false grade-0 that re-runs
# never revisited (`misses = work not in cache`) — silently re-introducing the
# pooling-bias holes the judge exists to eliminate. Transient errors now retry
# with exponential backoff; if still failing they are left UNCACHED (a miss) so
# a later run picks them up.
_JUDGE_MAX_RETRIES = 5
_JUDGE_BACKOFF_CAP_S = 30.0
# Substrings that mark a retryable transport/quota failure (Vertex + generic).
_TRANSIENT_MARKERS = (
    "429",
    "resource_exhausted",
    "503",
    "unavailable",
    "500 ",
    "internal",
    "504",
    "deadline",
    "timeout",
    "timed out",
    "connection",
)


def _is_transient(exc: Exception | None) -> bool:
    if exc is None:
        return False
    msg = str(exc).lower()
    return any(marker in msg for marker in _TRANSIENT_MARKERS)


# Auth/account-level failures are systemic: every subsequent call fails the same
# way, so degrading them to grade-0 would poison the entire remaining pool with
# false irrelevants. These ABORT the run loudly instead (matched against the
# EXCEPTION text only — never against a verdict's reason — so a judge that says
# "restricted area" in its reasoning is unaffected).
_FATAL_AUTH_MARKERS = (
    "invalid_grant",
    "account has been deleted",
    "account restricted",
    "access_denied",
    "permission_denied",
    "has been suspended",
    "unauthenticated",
    "401",
)


def _is_fatal_auth(exc: Exception | None) -> bool:
    if exc is None:
        return False
    msg = str(exc).lower()
    return any(marker in msg for marker in _FATAL_AUTH_MARKERS)


class JudgeAuthError(RuntimeError):
    """Raised to abort the whole judge run on a systemic auth/account failure."""

_RUBRIC = (
    "You are a relevance assessor for a graph-drawing / graph-layout literature "
    "search engine. Rate how well the DOCUMENT satisfies the information need "
    "behind the QUERY, on this 0-3 scale:\n"
    "  3 = Perfectly relevant: directly about the query's method/topic; a "
    "searcher wanting this query would want this document.\n"
    "  2 = Highly relevant: substantially on-topic and useful, even if not the "
    "single best match.\n"
    "  1 = Related: shares the broad area or some terminology but does not "
    "address the specific need.\n"
    "  0 = Irrelevant: different topic, or only incidental keyword overlap.\n"
    "Judge on substance, not keyword overlap. Reward conceptual/method matches "
    "even when wording differs; do not reward superficial term matches.\n"
    'Respond with ONLY a JSON object: {"grade": <0-3>, "reason": "<<=20 words>"}'
)


def judge_model() -> str:
    """Resolve the judge model name for the *active* LLM backend.

    The judge transport dispatches through ``rag_common.local_llm`` and honors
    ``RAG_LLM_BACKEND`` (``gemini`` default, or ``ollama``):

    * ``gemini``  → ``GRAPH_RAG_JUDGE_LLM_MODEL`` (the validated cloud floor).
    * ``ollama``  → ``RAG_OLLAMA_MODEL`` (a local model, runtime choice).

    Returning the backend-correct name is what keeps the model-aware judge cache
    key (``{model}:{case_id}:{doc_id}``) from colliding cloud and local grades.

    MODEL-SELECTION CONSTRAINT (rigor, not enforced in code — a runtime choice):
    the judge model MUST be **non-Qwen** — the embedder under test in this
    ladder is Qwen, and a Qwen judge would self-prefer Qwen-surfaced docs.
    It MUST also **differ from the HyDE / query-transform model** (also routed
    via ``local_llm``); reusing that model makes Phase-4 transform evaluation
    circular (the same model both writes the expansion and grades its hits).
    """
    from rag_common.local_llm import llm_backend, mlx_model, ollama_model, opencode_model

    backend = llm_backend()
    if backend == "ollama":
        return ollama_model()
    if backend == "mlx":
        return mlx_model()
    if backend == "opencode":
        return opencode_model()
    return os.getenv(JUDGE_MODEL_ENV, DEFAULT_JUDGE_MODEL).strip() or DEFAULT_JUDGE_MODEL


def _load_cache() -> dict[str, Any]:
    if not CACHE_PATH.is_file():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, sort_keys=True), encoding="utf-8")
    os.replace(tmp, CACHE_PATH)


def build_judge_prompt(query: str, doc: dict[str, Any]) -> str:
    """Render the source-blind judging prompt for one (query, doc) pair.

    Intentionally omits which systems surfaced the doc and any retrieval score.
    """
    title = doc.get("title") or "(untitled)"
    abstract = (doc.get("abstract") or "")[:MAX_ABSTRACT_CHARS]
    excerpt = (doc.get("excerpt") or "")[:MAX_EXCERPT_CHARS]
    parts = [
        _RUBRIC,
        "",
        f"QUERY: {query}",
        "",
        f"DOCUMENT TITLE: {title}",
    ]
    if abstract:
        parts.append(f"ABSTRACT: {abstract}")
    if excerpt:
        parts.append(f"MATCHED PASSAGE: {excerpt}")
    if not abstract and not excerpt:
        parts.append("(No abstract or passage available; judge from the title.)")
    return "\n".join(parts)


def _parse_grade(text: str) -> tuple[int, str]:
    raw = (text or "").strip()
    # Prefer a JSON object if present.
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            grade = int(obj.get("grade"))
            reason = str(obj.get("reason", ""))[:200]
            return max(0, min(3, grade)), reason
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    # Fallback: first standalone 0-3 digit.
    digit = re.search(r"\b([0-3])\b", raw)
    if digit:
        return int(digit.group(1)), raw[:200]
    return 0, raw[:200]


def _judge_one(query: str, doc: dict[str, Any], model: str) -> tuple[int, str]:
    """Grade one (query, doc) pair via the active ``local_llm`` backend.

    Transport only — the prompt and the 0-3 parsing are backend-identical. The
    Gemini path is preserved exactly (validated floor / fallback): a direct
    deterministic ``temperature=0.0`` client call. The Ollama path dispatches
    through ``rag_common.local_llm.generate_text`` at ``temperature=0.0``.
    """
    from rag_common.local_llm import llm_backend

    prompt = build_judge_prompt(query, doc)

    if llm_backend() in ("ollama", "mlx", "opencode"):
        from rag_common.local_llm import generate_text

        text = generate_text(prompt, model=model, temperature=0.0, max_tokens=128)
        return _parse_grade(text or "")

    # Gemini (default): keep the validated deterministic client call verbatim.
    from rag_common.gemini_embed import _client, llm_location

    client = _client(location=llm_location())
    config: Any = None
    try:  # deterministic if the genai types are importable
        from google.genai import types

        config = types.GenerateContentConfig(temperature=0.0)
    except Exception:  # noqa: BLE001
        config = None
    if config is not None:
        response = client.models.generate_content(model=model, contents=prompt, config=config)
    else:
        response = client.models.generate_content(model=model, contents=prompt)
    return _parse_grade(getattr(response, "text", None) or "")


def _judge_workers() -> int:
    raw = os.getenv("GRAPH_RAG_JUDGE_WORKERS", "8").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 8


def judge_pool(pool: dict[str, Any], *, model: str | None = None) -> dict[str, Any]:
    """Grade every pooled (query, doc) pair → qrels ``cases`` mapping.

    Returns ``{case_id: {doc_id: {grade, reason, systems, curated}}}``.
    """
    model = model or judge_model()
    cache = _load_cache()

    # Flatten to (cache_key, case_id, doc_id, query, doc) work items. The cache
    # key omits the track: a (query, doc) relevance grade is track-independent,
    # so judging the pdf track reuses the catalog grades for shared (case, doc).
    work: list[tuple[str, str, str, str, dict[str, Any]]] = []
    for case_id, case in pool["cases"].items():
        query = case["query"]
        for doc_id, doc in case["pooled"].items():
            key = f"{model}:{case_id}:{doc_id}"
            work.append((key, case_id, doc_id, query, doc))

    misses = [w for w in work if w[0] not in cache]

    def _run(item: tuple[str, str, str, str, dict[str, Any]]) -> tuple[str, dict[str, Any] | None]:
        key, _case_id, _doc_id, query, doc = item
        last_exc: Exception | None = None
        for attempt in range(_JUDGE_MAX_RETRIES + 1):
            try:
                grade, reason = _judge_one(query, doc, model)
                return key, {"grade": grade, "reason": reason}
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if _is_fatal_auth(exc):
                    # Systemic — abort now rather than poison the rest as grade-0.
                    raise JudgeAuthError(
                        f"aborting judge run on auth/account failure: {exc}"
                    ) from exc
                if not _is_transient(exc) or attempt == _JUDGE_MAX_RETRIES:
                    break
                # Exponential backoff + jitter so a quota burst self-throttles
                # instead of hammering Vertex (which only deepens the 429 wall).
                delay = min(_JUDGE_BACKOFF_CAP_S, 2.0**attempt) + random.uniform(0, 1.0)
                time.sleep(delay)
        # Permanent transient failure (e.g. quota exhausted after retries): return
        # None so the caller SKIPS caching — the pair stays a miss and a later run
        # retries it, rather than freezing a false grade-0 that re-runs never revisit.
        if _is_transient(last_exc):
            log.warning("judge transient-failed (not cached) for %s (%s)", key, last_exc)
            return key, None
        # Non-transient (malformed doc, parse error): degrade to 0 and cache — a
        # re-run would deterministically fail the same way, so caching is correct.
        log.warning("judge failed for %s (%s)", key, last_exc)
        return key, {"grade": 0, "reason": f"judge_error: {last_exc}"[:200]}

    if misses:
        done = 0
        skipped = 0
        with ThreadPoolExecutor(max_workers=_judge_workers()) as pool_exec:
            for key, verdict in pool_exec.map(_run, misses):
                done += 1
                if verdict is None:
                    skipped += 1
                else:
                    cache[key] = verdict
                if done % 200 == 0:
                    _save_cache(cache)
                    log.info("judged %d/%d (uncached transient fails: %d)", done, len(misses), skipped)
        _save_cache(cache)
        log.info(
            "judged %d new (query, doc) pairs; %d left uncached (transient — rerun to retry)",
            len(misses) - skipped,
            skipped,
        )

    cases_out: dict[str, dict[str, dict[str, Any]]] = {}
    for key, case_id, doc_id, _query, doc in work:
        verdict = cache.get(key, {"grade": 0, "reason": ""})
        cases_out.setdefault(case_id, {})[doc_id] = {
            "grade": int(verdict.get("grade", 0)),
            "reason": verdict.get("reason", ""),
            "systems": sorted(doc.get("systems", {})),
            "curated": bool(doc.get("curated")),
        }
    return cases_out


HG7_MIN_SPEARMAN = 0.70
HG7_MIN_KAPPA = 0.70
HG7_MAX_BIAS = 0.25


def _hg7_decide(spearman: float, kappa: float, bias: float) -> bool:
    """HG7 PASS iff rho>=0.70 AND kappa>=0.70 AND bias<0.25."""
    return spearman >= HG7_MIN_SPEARMAN and kappa >= HG7_MIN_KAPPA and bias < HG7_MAX_BIAS


def _load_analyze_bakeoff() -> Any:
    """Load ``scripts/analyze_bakeoff.py`` (a script, not a package module)."""
    import importlib.util

    audit_path = Path(__file__).resolve().parents[3] / "scripts" / "analyze_bakeoff.py"
    spec = importlib.util.spec_from_file_location("graph_rag_analyze_bakeoff", audit_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load analyze_bakeoff audit from {audit_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_judge_hg7(
    pool_path: str | Path,
    *,
    model: str | None = None,
    seed_cases_path: str | Path | None = None,
) -> dict[str, Any]:
    """HG7 judge-validation gate — run BEFORE the judge's grades feed qrels.

    Grades the pool (so curated/seed docs are scored by the *active* judge
    backend), writes the graded qrels, then reuses
    ``scripts/analyze_bakeoff.py:judge_audit`` to compute agreement against the
    curated labels: Spearman rho, Cohen kappa, and the dense-vs-BM25 "bias"
    gap. Returns ``PASS`` iff ``rho>=0.70 AND kappa>=0.70 AND bias<0.25``.

    The judge model is resolved via :func:`judge_model` (backend-correct), so a
    Qwen embedder under test never validates its own judge — see the
    model-selection constraint documented on :func:`judge_model`.

    Curated labels come from the pooled docs' ``curated`` flag (the curated-49
    gold set), surfaced as ``curated=True`` in the graded qrels; ``judge_audit``
    treats those as relevance-3 seeds.
    """
    import tempfile

    from graph_layout_rag.eval.qrels import DEFAULT_RELEVANCE_THRESHOLD, write_qrels

    resolved_model = model or judge_model()
    pool = json.loads(Path(pool_path).read_text(encoding="utf-8"))
    cases_out = judge_pool(pool, model=resolved_model)

    module = _load_analyze_bakeoff()

    with tempfile.TemporaryDirectory() as tmp:
        qrels_path = Path(tmp) / "qrels.json"
        write_qrels(
            qrels_path,
            cases_out,
            judge_model=resolved_model,
            relevance_threshold=DEFAULT_RELEVANCE_THRESHOLD,
            extra={"track": pool.get("track", "catalog")},
        )
        audit = module.judge_audit(
            pool_path=str(Path(pool_path)),
            qrels_path=str(qrels_path),
            seed_cases_path=str(seed_cases_path) if seed_cases_path else None,
        )

    spearman = float(audit.get("seed_spearman", 0.0))
    kappa = float(audit.get("seed_kappa", 0.0))
    # Corrected bias metric: the source-blind judge can only be provenance-biased
    # if it disagrees with curated GROUND TRUTH at different rates by provenance.
    # (The old metric was dense_rel_rate - bm25_rel_rate, which mistook the
    # de-biased pool's payoff — dense surfacing relevant docs BM25 missed — for
    # judge bias.) Two-sided gap: bias in either direction trips the gate.
    bias = abs(
        float(audit.get("dense_disagree_rate", 0.0))
        - float(audit.get("bm25_disagree_rate", 0.0))
    )
    passed = _hg7_decide(spearman, kappa, bias)
    return {
        "gate": "PASS" if passed else "FAIL",
        "passed": passed,
        "model": resolved_model,
        "spearman": round(spearman, 4),
        "kappa": round(kappa, 4),
        "bias": round(bias, 4),
        "thresholds": {
            "min_spearman": HG7_MIN_SPEARMAN,
            "min_kappa": HG7_MIN_KAPPA,
            "max_bias": HG7_MAX_BIAS,
        },
        "audit": audit,
    }


def judge_agreement(cases_out: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """Free quality check: how the judge graded the curated (known-relevant) docs.

    A well-calibrated judge should grade most curated docs >= 2. A low rate flags
    either judge miscalibration or genuinely mislabeled curated docs to review.
    """
    grades: list[int] = []
    for docs in cases_out.values():
        for verdict in docs.values():
            if verdict.get("curated"):
                grades.append(int(verdict.get("grade", 0)))
    if not grades:
        return {"curated_judged": 0}
    relevant = sum(1 for g in grades if g >= 2)
    return {
        "curated_judged": len(grades),
        "curated_grade>=2_rate": round(relevant / len(grades), 3),
        "curated_grade_mean": round(sum(grades) / len(grades), 3),
        "curated_grade_hist": {g: grades.count(g) for g in (0, 1, 2, 3)},
    }
