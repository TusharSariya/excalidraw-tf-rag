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
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from rag_literature_rag.paths import DATA_DIR

log = logging.getLogger("rag_literature_rag.eval.judge")

CACHE_PATH = DATA_DIR / "eval" / "judge_cache.json"
MAX_ABSTRACT_CHARS = 1500
MAX_EXCERPT_CHARS = 800

# Domain rubric version — included in the judge cache key so that changing the
# domain framing (or the scale wording) invalidates stale grades instead of
# silently reusing them. Bump when _RUBRIC's meaning changes.
_RUBRIC_VERSION = "rag-v2"

_RUBRIC = (
    "You are a relevance assessor for a retrieval-augmented generation (RAG) "
    "literature search engine. Documents are research papers on RAG and neural "
    "information retrieval (e.g. dense/sparse retrieval, hybrid search, "
    "reranking and late interaction, GraphRAG, agentic and self-correcting RAG, "
    "query rewriting, chunking, and RAG evaluation). Rate how well the DOCUMENT "
    "satisfies the information need behind the QUERY, on this 0-3 scale:\n"
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
    """Resolve the judge model name for the *active* judge backend.

    Default (``RAG_LIT_JUDGE_BACKEND`` unset): the direct-HTTP dual backend
    (DeepSeek API primary, opencode-go Zen fallback on credit/rate-limit
    errors) — see ``_http_judge_backend``/``_backend_order`` below. Set
    ``RAG_LIT_JUDGE_BACKEND=local`` (or ``ollama``) to opt back into the
    ``rag_common.local_llm`` dispatch instead. Bulk judging must NEVER route
    through the opencode CLI subprocess path — only the HTTP Zen endpoint or
    the DeepSeek API.

    The *actual* model that produced a given grade (which may differ from this
    resolved name when the HTTP dual backend fails over) is recorded per-case
    in the qrels entry's ``judge_model`` field — see ``judge_pool``.
    """
    http_backend = _http_judge_backend()
    if http_backend in ("opencode", "auto"):
        return os.getenv("RAG_LIT_JUDGE_HTTP_MODEL", "opencode-go/deepseek-v4-flash")
    if http_backend == "deepseek":
        return os.getenv("RAG_LIT_DEEPSEEK_JUDGE_MODEL", "deepseek-chat")

    from rag_common.local_llm import llm_backend, ollama_model, opencode_model

    if llm_backend() == "opencode":
        return opencode_model()
    return ollama_model()


# --- Direct-HTTP dual-backend judge (DeepSeek API + opencode-go Zen) -------
# Bypasses rag_common.local_llm's opencode-CLI path for the OpenAI-compatible
# HTTP endpoints, so a large judging run gets (a) real concurrency (judge_pool
# owns its threads) and (b) automatic FAILOVER when one backend runs out of
# credits. This is the DEFAULT judge path (RAG_LIT_JUDGE_BACKEND unset ==
# "auto"): DeepSeek API primary, opencode-go Zen fallback on credit/rate-limit
# errors. Override via RAG_LIT_JUDGE_BACKEND:
#   "deepseek" -> DeepSeek API only ; "opencode" -> opencode-go only ;
#   "auto"     -> DeepSeek primary, fail over to opencode-go on credit
#                 exhaustion / rate limit (the default) ;
#   "local"/"ollama"/"none"/"off" -> opt OUT of the HTTP path entirely and use
#                 the rag_common.local_llm dispatch below (Ollama/opencode
#                 CLI/gateway) — bulk judging should not need this.
_OPENCODE_URL = "https://opencode.ai/zen/go/v1/chat/completions"
# A non-empty User-Agent is REQUIRED — the gateway's Cloudflare returns 403 to
# urllib's default UA. (Discovered 2026-06-27.)
_OPENCODE_UA = "opencode/1.17.11"
_DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
_OPENCODE_RETRYABLE_HTTP = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_HTTP_CREDIT = frozenset({401, 402, 403})  # auth / insufficient-balance -> fail over, don't retry
_HTTP_DISABLE = frozenset({"local", "ollama", "none", "off", ""})


class _JudgeCreditError(RuntimeError):
    """A backend refused for auth/credit reasons (401/402/403) — try the next one."""


def _http_judge_backend() -> str:
    """RAG_LIT_JUDGE_BACKEND, defaulting to "auto" (DeepSeek primary, opencode
    fallback).

    Deliberately does NOT infer from an "opencode-go/..." model string: the
    resolved default judge model itself looks like an opencode-go id (see
    judge_model()), so inferring backend from that string would silently
    collapse the default "auto" dual-backend into "opencode"-only.
    """
    return os.getenv("RAG_LIT_JUDGE_BACKEND", "auto").strip().lower()


def _backend_order(backend: str) -> list[str]:
    if backend == "auto":
        return ["deepseek", "opencode"]
    if backend in ("opencode", "deepseek"):
        return [backend]
    return []


def _opencode_key() -> str:
    key = os.getenv("OPENCODE_API_KEY", "").strip()
    if key:
        return key
    try:
        auth = json.loads(
            (Path.home() / ".local/share/opencode/auth.json").read_text(encoding="utf-8")
        )
        return str((auth.get("opencode-go") or {}).get("key", "")).strip()
    except (OSError, json.JSONDecodeError, AttributeError):
        return ""


def _deepseek_key() -> str:
    return os.getenv("DEEPSEEK_API_KEY", "").strip()


def _http_chat_grade(
    url: str, key: str, api_model: str, prompt: str, *, user_agent: str | None = None
) -> tuple[int, str]:
    """POST one OpenAI-compatible chat completion and parse a 0-3 grade.

    Retries transient HTTP/timeout with jittered backoff; raises
    ``_JudgeCreditError`` on 401/402/403 (credit/auth → caller fails over); raises
    the last exception on persistent transient failure (so judge_pool records a
    non-cached judge_error, retried next run, rather than a fake grade-0).
    """
    payload = json.dumps(
        {
            "model": api_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 256,
        }
    ).encode()
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if user_agent:
        headers["User-Agent"] = user_agent

    last_exc: Exception | None = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.load(resp)
            text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            if not text.strip():
                raise ValueError("empty judge response")
            return _parse_grade(text)
        except urllib.error.HTTPError as exc:  # noqa: PERF203
            last_exc = exc
            if exc.code in _HTTP_CREDIT:
                raise _JudgeCreditError(f"backend refused ({exc.code})") from exc
            if exc.code in _OPENCODE_RETRYABLE_HTTP and attempt < 4:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                base = (
                    float(retry_after)
                    if (retry_after and str(retry_after).isdigit())
                    else min(30.0, 2**attempt)
                )
                time.sleep(base + random.uniform(0, max(0.5, base * 0.25)))
                continue
            raise
        except _JudgeCreditError:
            raise
        except Exception as exc:  # noqa: BLE001 — timeout / empty / decode → retry then raise
            last_exc = exc
            if attempt < 4:
                delay = min(30.0, 1.5 * (2**attempt))
                time.sleep(delay + random.uniform(0, 1.0))
                continue
            raise
    raise last_exc or RuntimeError("http judge failed")


def _judge_one_opencode(query: str, doc: dict[str, Any], model: str) -> tuple[int, str, str]:
    """Grade one (query, doc) via the OpenCode Go OpenAI-compatible endpoint.

    Resilient by contract: retries 429/5xx/timeout/empty with jittered backoff,
    raises ``_JudgeCreditError`` on 401/402/403 (caller fails over), and RAISES
    on persistent failure so judge_pool records a non-cached judge_error
    (retried next run) rather than poisoning qrels with a fake 0.
    """
    key = _opencode_key()
    if not key:
        raise _JudgeCreditError("opencode: no OPENCODE_API_KEY and no auth.json opencode-go key")
    api_model = model.split("/", 1)[1] if "/" in model else model
    grade, reason = _http_chat_grade(
        _OPENCODE_URL, key, api_model, build_judge_prompt(query, doc), user_agent=_OPENCODE_UA
    )
    return grade, reason, f"opencode-go/{api_model}"


def _judge_one_deepseek(query: str, doc: dict[str, Any], model: str) -> tuple[int, str, str]:
    key = _deepseek_key()
    if not key:
        raise _JudgeCreditError("deepseek: DEEPSEEK_API_KEY unset")
    # The Zen id (deepseek-v4-flash) is not a DeepSeek-API model name; map to a
    # configurable direct-API model (default deepseek-chat).
    api_model = os.getenv("RAG_LIT_DEEPSEEK_JUDGE_MODEL", "deepseek-chat")
    grade, reason = _http_chat_grade(_DEEPSEEK_URL, key, api_model, build_judge_prompt(query, doc))
    return grade, reason, f"deepseek:{api_model}"


_HTTP_JUDGES = {"opencode": _judge_one_opencode, "deepseek": _judge_one_deepseek}


def _judge_one_http(
    query: str, doc: dict[str, Any], model: str, backend: str
) -> tuple[int, str, str]:
    """Grade via the HTTP backends in order, failing over on credit/other errors.

    Returns ``(grade, reason, actual_model)`` — ``actual_model`` names the
    backend/model that actually produced the grade (not just the requested
    ``model``), for per-grade auditability even when a failover occurred.
    """
    last_exc: Exception | None = None
    for name in _backend_order(backend):
        try:
            return _HTTP_JUDGES[name](query, doc, model)
        except _JudgeCreditError as exc:
            log.warning("judge backend %s refused/exhausted (%s); failing over", name, exc)
            last_exc = exc
        except Exception as exc:  # noqa: BLE001 — one backend down; try the next before giving up
            log.warning("judge backend %s failed (%s); failing over", name, exc)
            last_exc = exc
    raise last_exc or RuntimeError(f"no usable judge backend for {backend!r}")


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


def _judge_one(query: str, doc: dict[str, Any], model: str) -> tuple[int, str, str]:
    """Grade one (query, doc) pair via the active judge backend.

    Returns ``(grade, reason, actual_model)``. Defaults to the direct-HTTP
    dual backend (DeepSeek primary, opencode-go fallback); set
    ``RAG_LIT_JUDGE_BACKEND=local``/``ollama`` to dispatch through
    ``rag_common.local_llm.generate_text`` instead (never used for bulk
    judging).
    """
    http_backend = _http_judge_backend()
    if http_backend not in _HTTP_DISABLE:
        return _judge_one_http(query, doc, model, http_backend)

    from rag_common.local_llm import generate_text

    prompt = build_judge_prompt(query, doc)
    text = generate_text(prompt, model=model, temperature=0.0, max_tokens=256)
    grade, reason = _parse_grade(text or "")
    return grade, reason, model


def _judge_workers() -> int:
    raw = os.getenv("RAG_LIT_JUDGE_WORKERS", "8").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 8


def judge_pool(pool: dict[str, Any], *, model: str | None = None) -> dict[str, Any]:
    """Grade every pooled (query, doc) pair → qrels ``cases`` mapping.

    Returns ``{case_id: {doc_id: {grade, reason, systems, curated, judge_model}}}``.
    ``judge_model`` is the actual model/backend that produced that particular
    grade (may differ from the file-level ``judge_model`` in ``write_qrels``
    when the HTTP dual backend failed over mid-run).
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
            key = f"{model}:{_RUBRIC_VERSION}:{case_id}:{doc_id}"
            work.append((key, case_id, doc_id, query, doc))

    misses = [w for w in work if w[0] not in cache]

    def _run(item: tuple[str, str, str, str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        key, _case_id, _doc_id, query, doc = item
        try:
            grade, reason, actual_model = _judge_one(query, doc, model)
        except Exception as exc:  # noqa: BLE001 — degrade to 0, keep going
            log.warning("judge failed for %s (%s)", key, exc)
            return key, {"grade": 0, "reason": f"judge_error: {exc}"[:200], "judge_model": model}
        return key, {"grade": grade, "reason": reason, "judge_model": actual_model}

    if misses:
        done = cached = 0
        with ThreadPoolExecutor(max_workers=_judge_workers()) as pool_exec:
            for key, verdict in pool_exec.map(_run, misses):
                # Do not persist error verdicts: a poisoned grade-0 would silently
                # contaminate qrels. Leaving them out means they retry next run.
                if "judge_error" not in str(verdict.get("reason", "")):
                    cache[key] = verdict
                    cached += 1
                done += 1
                if done % 200 == 0:
                    _save_cache(cache)
                    log.info("judged %d/%d (%d ok)", done, len(misses), cached)
        _save_cache(cache)
        log.info("judged %d new pairs (%d ok, %d errored→will retry)", done, cached, done - cached)

    cases_out: dict[str, dict[str, dict[str, Any]]] = {}
    for key, case_id, doc_id, _query, doc in work:
        verdict = cache.get(key, {"grade": 0, "reason": ""})
        entry: dict[str, Any] = {
            "grade": int(verdict.get("grade", 0)),
            "reason": verdict.get("reason", ""),
            "systems": sorted(doc.get("systems", {})),
            "curated": bool(doc.get("curated")),
        }
        # Per-grade judge-model auditability: record the ACTUAL model/backend
        # that produced this grade (may differ from the file-level judge_model
        # in write_qrels() if the HTTP dual backend failed over mid-run).
        # Absent for entries loaded from legacy cache rows written before this
        # field existed — treat a missing field as "the file-level judge_model
        # applies" (see qrels.py docstring).
        actual_model = verdict.get("judge_model")
        if actual_model:
            entry["judge_model"] = actual_model
        cases_out.setdefault(case_id, {})[doc_id] = entry
    return cases_out


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
