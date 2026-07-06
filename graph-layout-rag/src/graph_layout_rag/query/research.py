"""Deep-research pipeline: `research <question>` — the agentic synthesis arm.

Extends the PaperQA2 / Search-o1 loop in ``agent.py`` into a full answer-writing
command:

    decompose  -> <=4 focused sub-questions (1 cheap LLM call)
    gather     -> hybrid retrieval top-N per sub-question (reuses ``search``)
    expand     -> citation following: for the top docs of each sub-question, pull
                  1-hop in-corpus cite-neighbors and keep any that clear an embed
                  similarity bar (no LLM — the cheap primitive)
    summarize  -> one short evidence summary per unique doc (direct-HTTP DeepSeek)
    synthesize -> a cited answer + references (one larger LLM call), with a
                  structured ``SUFFICIENT: yes/no`` self-report
    iterate    -> at most ONE follow-up round when synthesis reports insufficiency
                  (bounded by ``depth``)

Two LLM lanes, both env-overridable (point 4 of the P2.4 spec):

* **cheap** steps (decompose, follow-up) resolve through ``rag_common.local_llm``
  (Ollama/opencode/gateway) by default — ``GRAPH_RAG_RESEARCH_CHEAP_BACKEND``.
* **synth** steps (evidence summaries, final synthesis) default to the direct-HTTP
  DeepSeek helper (``deepseek-chat``) modeled on ``eval.judge._http_chat_grade`` —
  ``GRAPH_RAG_RESEARCH_SYNTH_BACKEND``. Set either to ``deepseek``/``opencode``/``local``.

``--dry-run`` runs retrieval + citation expansion only and makes ZERO LLM calls.
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
from dataclasses import dataclass, field
from typing import Any

from graph_layout_rag.query.search import search

log = logging.getLogger("graph_layout_rag.query.research")

MAX_SUBQUESTIONS = 4
EVIDENCE_DOC_CAP = 12
CITE_NEIGHBOR_LIMIT = 8
CITE_SIM_THRESHOLD = 0.30
DEFAULT_TOP = 8

# Rough DeepSeek deepseek-chat pricing (USD per 1M tokens) for the cost readout.
# Not billing-accurate — a transparency estimate the user can sanity-check.
_USD_PER_1M_INPUT = 0.27
_USD_PER_1M_OUTPUT = 1.10


# --------------------------------------------------------------------------- #
# Cost accounting
# --------------------------------------------------------------------------- #
def _est_tokens(text: str) -> int:
    """~4 chars/token heuristic; good enough for a transparency estimate."""
    return max(1, len(text or "") // 4)


@dataclass
class Budget:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    by_step: dict[str, int] = field(default_factory=dict)

    def record(self, step: str, prompt: str, output: str) -> None:
        self.calls += 1
        self.input_tokens += _est_tokens(prompt)
        self.output_tokens += _est_tokens(output)
        self.by_step[step] = self.by_step.get(step, 0) + 1

    def usd_estimate(self) -> float:
        return round(
            self.input_tokens / 1_000_000 * _USD_PER_1M_INPUT
            + self.output_tokens / 1_000_000 * _USD_PER_1M_OUTPUT,
            4,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "llm_calls": self.calls,
            "calls_by_step": dict(self.by_step),
            "input_tokens_est": self.input_tokens,
            "output_tokens_est": self.output_tokens,
            "usd_est": self.usd_estimate(),
        }


# --------------------------------------------------------------------------- #
# LLM lanes (cheap = local_llm; synth = direct-HTTP DeepSeek)
# --------------------------------------------------------------------------- #
_SYNTH_STEPS = frozenset({"evidence", "synthesis"})
_DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
_OPENCODE_URL = "https://opencode.ai/zen/go/v1/chat/completions"
_OPENCODE_UA = "opencode/1.17.11"
_RETRYABLE_HTTP = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def _backend_for(step: str) -> str:
    if step in _SYNTH_STEPS:
        return os.getenv("GRAPH_RAG_RESEARCH_SYNTH_BACKEND", "deepseek").strip().lower()
    return os.getenv("GRAPH_RAG_RESEARCH_CHEAP_BACKEND", "local").strip().lower()


def _http_chat(
    url: str, key: str, model: str, prompt: str, *, max_tokens: int, ua: str | None
) -> str:
    """POST one OpenAI-compatible chat completion, return the text. Retries transient."""
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
    ).encode()
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if ua:
        headers["User-Agent"] = ua
    last_exc: Exception | None = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.load(resp)
            text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            if not text.strip():
                raise ValueError("empty response")
            return text.strip()
        except urllib.error.HTTPError as exc:  # noqa: PERF203
            last_exc = exc
            if exc.code in _RETRYABLE_HTTP and attempt < 4:
                time.sleep(min(20.0, 2**attempt) + random.uniform(0, 1.0))
                continue
            raise
        except Exception as exc:  # noqa: BLE001 — timeout/empty/decode → retry then raise
            last_exc = exc
            if attempt < 4:
                time.sleep(1.5 * (2**attempt) + random.uniform(0, 1.0))
                continue
            raise
    raise last_exc or RuntimeError("http chat failed")


def _http_generate(prompt: str, *, backend: str, max_tokens: int) -> str:
    from graph_layout_rag.eval.judge import _deepseek_key, _opencode_key

    if backend == "deepseek":
        key = _deepseek_key()
        if not key:
            raise RuntimeError(
                "GRAPH_RAG_RESEARCH_SYNTH_BACKEND=deepseek but DEEPSEEK_API_KEY unset"
            )
        model = os.getenv("GRAPH_RAG_DEEPSEEK_JUDGE_MODEL", "deepseek-chat")
        return _http_chat(_DEEPSEEK_URL, key, model, prompt, max_tokens=max_tokens, ua=None)
    if backend == "opencode":
        key = _opencode_key()
        if not key:
            raise RuntimeError("research opencode backend but no OPENCODE_API_KEY / auth.json key")
        model = os.getenv("GRAPH_RAG_JUDGE_HTTP_MODEL", "opencode-go/deepseek-v4-flash")
        api_model = model.split("/", 1)[1] if "/" in model else model
        return _http_chat(
            _OPENCODE_URL, key, api_model, prompt, max_tokens=max_tokens, ua=_OPENCODE_UA
        )
    raise RuntimeError(f"unknown HTTP research backend {backend!r}")


def generate(prompt: str, *, step: str, max_tokens: int, budget: Budget) -> str:
    """Single generation entry point (monkeypatched wholesale in tests)."""
    backend = _backend_for(step)
    if backend in ("deepseek", "opencode"):
        text = _http_generate(prompt, backend=backend, max_tokens=max_tokens)
    else:
        from rag_common.local_llm import generate_text

        text = generate_text(prompt, max_tokens=max_tokens) or ""
    budget.record(step, prompt, text)
    return text


# --------------------------------------------------------------------------- #
# Pipeline stages (all monkeypatched by name in tests)
# --------------------------------------------------------------------------- #
def decompose(question: str, *, budget: Budget, max_sub: int = MAX_SUBQUESTIONS) -> list[str]:
    """Break the question into <=max_sub focused sub-questions (1 cheap LLM call)."""
    prompt = (
        "Break this graph drawing / layout research question into "
        f"{max_sub} or fewer focused sub-questions that, answered together, cover it. "
        "Use precise algorithm/technique terms (e.g. Sugiyama layering, stress majorization, "
        "crossing minimization, orthogonal routing). One sub-question per line, no numbering.\n\n"
        f"Question: {question}"
    )
    try:
        raw = generate(prompt, step="decompose", max_tokens=256, budget=budget)
    except Exception as exc:  # noqa: BLE001 — degrade to the bare question
        log.warning("research decompose failed (%s)", exc)
        return []
    subs = [re.sub(r"^\s*[\d.)\-*\s]+", "", ln).strip() for ln in raw.splitlines()]
    return [s for s in subs if s][:max_sub]


def _followup(question: str, titles: list[str], *, budget: Budget) -> str | None:
    listing = "\n".join(f"- {t}" for t in titles[:12])
    prompt = (
        "You are gathering papers to answer a graph-layout research question.\n"
        f"Question: {question}\n\nRetrieved so far:\n{listing}\n\n"
        "If these cover the question, reply exactly DONE. Otherwise reply with ONE "
        "additional search query (precise terms) for the missing aspect."
    )
    try:
        reply = generate(prompt, step="followup", max_tokens=64, budget=budget).strip()
    except Exception as exc:  # noqa: BLE001
        log.warning("research followup failed (%s)", exc)
        return None
    if not reply or reply.upper().startswith("DONE"):
        return None
    return re.sub(r"^(query[:\-\s]*)", "", reply, flags=re.IGNORECASE).strip() or None


def _gather(
    subquestions: list[str],
    *,
    top: int,
    embed_profile: str | None,
    rerank: bool | None,
    hybrid: bool,
    filters: dict[str, Any],
    graph_arm: Any | None,
) -> list[dict[str, Any]]:
    """Retrieve per sub-question and fuse to a doc-level pool (no LLM)."""
    pooled: dict[str, dict[str, Any]] = {}
    for sq in subquestions:
        results: list[dict[str, Any]] | None = None
        if graph_arm is not None:
            try:
                results = graph_arm.gather(  # soft contract; see _resolve_graph_arm
                    sq, top=top, embed_profile=embed_profile, filters=filters
                )
            except Exception as exc:  # noqa: BLE001 — never let the optional arm break retrieval
                log.warning("graph arm gather failed (%s); falling back to hybrid", exc)
                results = None
        if results is None:
            results = search(
                sq,
                top=top,
                embed_profile=embed_profile,
                rerank=rerank,
                hybrid=hybrid,
                max_per_doc=2,
                **filters,
            )
        for r in results:
            key = r.get("canonical_doc_id") or r.get("doc_id") or ""
            if not key:
                continue
            prev = pooled.get(key)
            if prev is None or float(r.get("score") or 0) > float(prev.get("score") or 0):
                pooled[key] = r
    ranked = sorted(pooled.values(), key=lambda r: float(r.get("score") or 0), reverse=True)
    return ranked


# --------------------------------------------------------------------------- #
# Citation-following expansion (1-hop in-corpus neighbors, embed-scored)
# --------------------------------------------------------------------------- #
def _one_hop_neighbor_docs(seed_doc_ids: list[str], *, limit: int) -> list[str]:
    """In-corpus doc_ids one citation hop from the seeds (builds-on + cited-by).

    Best-effort: returns [] when there is no citation store. Sorted by global
    cited-by-count so the most load-bearing neighbors come first.
    """
    from graph_layout_rag.query.citation_rank import load_graph_cached

    try:
        g = load_graph_cached()
    except Exception as exc:  # noqa: BLE001
        log.warning("citation graph load failed (%s)", exc)
        return []
    if g is None:
        return []
    seen: dict[str, int] = {}
    for doc_id in seed_doc_ids:
        oa = g.doc_to_oa.get(doc_id)
        if oa is None:
            continue
        for nb in g.out_adj.get(oa, set()) | g.in_adj.get(oa, set()):
            if nb == oa:
                continue
            nb_doc = g.oa_to_doc.get(nb)
            if nb_doc and nb_doc not in seed_doc_ids:
                seen[nb_doc] = max(seen.get(nb_doc, 0), g.cbc.get(nb, 0))
    return [d for d, _ in sorted(seen.items(), key=lambda kv: kv[1], reverse=True)][:limit]


def _neighbor_similarity(
    doc_id: str, vectors: list[list[float]], *, embed_profile: str | None
) -> dict[str, Any] | None:
    """Best chunk of ``doc_id`` vs any sub-question vector (cosine). None if unindexed."""
    from graph_layout_rag.query.retrieve import resolve_retrieve_context

    ctx = resolve_retrieve_context(embed_profile=embed_profile)
    best: dict[str, Any] | None = None
    safe_id = doc_id.replace("'", "''")
    for vec in vectors:
        try:
            rows = (
                ctx.table.search(list(vec))
                .metric("cosine")
                .where(f"doc_id = '{safe_id}'", prefilter=True)
                .limit(1)
                .to_list()
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("neighbor similarity query failed for %s (%s)", doc_id, exc)
            return None
        if not rows:
            continue
        row = rows[0]
        distance = row.get("_distance")
        sim = 1.0 - float(distance) if distance is not None else 0.0
        if best is None or sim > best["score"]:
            row.pop("vector", None)
            best = {"row": row, "score": sim}
    return best


def expand_citations(
    pool: list[dict[str, Any]],
    subquestions: list[str],
    *,
    top_docs: int,
    embed_profile: str | None,
    threshold: float = CITE_SIM_THRESHOLD,
    cap: int,
) -> list[dict[str, Any]]:
    """Add citation-neighbor docs that clear the embed-similarity bar (no LLM).

    Seeds are the top ``top_docs`` of the current pool. Returns the extra entries
    (search-result shape, ``via_citation=True``) to append — never mutates ``pool``.
    """
    if not pool or cap <= 0:
        return []
    from graph_layout_rag.query.identity import canonical_identity_map
    from graph_layout_rag.query.retrieve import _embed_vector, resolve_retrieve_context

    seed_ids: list[str] = []
    for r in pool[:top_docs]:
        seed_ids.extend(r.get("alias_doc_ids") or [])
        if r.get("doc_id"):
            seed_ids.append(r["doc_id"])
        if r.get("canonical_doc_id"):
            seed_ids.append(r["canonical_doc_id"])
    neighbors = _one_hop_neighbor_docs(seed_ids, limit=CITE_NEIGHBOR_LIMIT)
    if not neighbors:
        return []

    present = {r.get("canonical_doc_id") or r.get("doc_id") for r in pool}
    identities = canonical_identity_map()
    try:
        ctx = resolve_retrieve_context(embed_profile=embed_profile)
        vectors = [_embed_vector(sq, config=ctx.config) for sq in subquestions]
    except Exception as exc:  # noqa: BLE001 — no index/embedder → skip expansion
        log.warning("citation expansion embed setup failed (%s)", exc)
        return []

    return _select_neighbors(
        neighbors,
        present,
        canonicalize=identities.canonicalize_doc_id,
        similarity=lambda doc_id: _neighbor_similarity(
            doc_id, vectors, embed_profile=embed_profile
        ),
        threshold=threshold,
        cap=cap,
    )


def _select_neighbors(
    neighbors: list[str],
    present: set[str | None],
    *,
    canonicalize: Any,
    similarity: Any,
    threshold: float,
    cap: int,
) -> list[dict[str, Any]]:
    """Pure inclusion logic: keep neighbors that (de-dup) clear ``threshold``, up to ``cap``.

    ``similarity(doc_id)`` returns ``{"row": <chunk>, "score": float}`` or None. Split out
    from ``expand_citations`` so the threshold / cap / de-dup rules are unit-testable
    without an index or embedder.
    """
    added: list[dict[str, Any]] = []
    for doc_id in neighbors:
        canonical = canonicalize(doc_id)
        if canonical in present:
            continue
        hit = similarity(doc_id)
        if hit is None or hit["score"] < threshold:
            continue
        row = hit["row"]
        text = row.get("text") or ""
        excerpt = text[:400] + ("..." if len(text) > 400 else "")
        added.append(
            {
                "score": round(hit["score"], 4),
                "title": row.get("title"),
                "excerpt": excerpt,
                "source_url": row.get("source_url"),
                "page": row.get("page"),
                "page_end": row.get("page_end"),
                "doc_id": row.get("doc_id") or doc_id,
                "canonical_doc_id": canonical,
                "section_path": row.get("section_path") or "",
                "via_citation": True,
                "evidence": [
                    {
                        "excerpt": excerpt,
                        "page": row.get("page"),
                        "section_path": row.get("section_path") or "",
                        "score": round(hit["score"], 4),
                    }
                ],
            }
        )
        present.add(canonical)
        if len(added) >= cap:
            break
    return added


# --------------------------------------------------------------------------- #
# References / BibTeX keys
# --------------------------------------------------------------------------- #
def _ascii_fold(text: str) -> str:
    import unicodedata

    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Za-z0-9]+", "", folded)


def _bib_key(authors: list[str], year: int | None, title: str | None, fallback: str) -> str:
    surname = ""
    if authors:
        name = authors[0].strip()
        surname = name.split(",", 1)[0] if "," in name else (name.split()[-1] if name else "")
    surname_tok = _ascii_fold(surname)
    year_tok = str(year) if year is not None else ""
    title_tok = _ascii_fold(title.split()[0]) if title and title.split() else ""
    key = f"{surname_tok}{year_tok}{title_tok}".lower()
    return key or _ascii_fold(fallback).lower() or "ref"


def _manifest_index() -> dict[str, Any]:
    from graph_layout_rag.manifest import load_manifest

    idx: dict[str, Any] = {}
    for item in load_manifest().items:
        idx[item.id] = item
        for alias in getattr(item, "aliasDocIds", []) or []:
            idx.setdefault(alias, item)
    return idx


def _reference_for(
    doc_id: str, canonical_id: str, title: str | None, manifest: dict[str, Any]
) -> dict[str, Any]:
    item = manifest.get(canonical_id) or manifest.get(doc_id)
    authors = list(getattr(item, "authors", []) or []) if item else []
    year = getattr(item, "year", None) if item else None
    ref_title = (getattr(item, "title", None) if item else None) or title or canonical_id
    return {
        "doc_id": canonical_id,
        "bib_key": _bib_key(authors, year, ref_title, canonical_id),
        "title": ref_title,
        "year": year,
        "authors": authors[:6],
    }


# --------------------------------------------------------------------------- #
# Evidence summarization + synthesis
# --------------------------------------------------------------------------- #
def _doc_context(doc: dict[str, Any]) -> str:
    parts: list[str] = []
    for ev in (doc.get("evidence") or [])[:2]:
        page = f" (p.{ev.get('page')})" if ev.get("page") else ""
        parts.append(f"{ev.get('excerpt', '')}{page}")
    if not parts and doc.get("excerpt"):
        parts.append(doc["excerpt"])
    return "\n---\n".join(parts)


def summarize_evidence(doc: dict[str, Any], question: str, *, budget: Budget) -> str:
    context = _doc_context(doc)
    prompt = (
        "In 2-3 sentences, state what this paper's passages contribute to answering the "
        "question. Be specific about methods/findings; if the passages are not relevant, "
        'say "not relevant".\n\n'
        f"Question: {question}\n\nPaper: {doc.get('title') or doc.get('doc_id')}\n\n"
        f"Passages:\n{context}"
    )
    try:
        return generate(prompt, step="evidence", max_tokens=200, budget=budget).strip()
    except Exception as exc:  # noqa: BLE001 — a failed summary shouldn't sink the run
        log.warning("evidence summary failed for %s (%s)", doc.get("doc_id"), exc)
        return ""


_SUFFICIENT_RE = re.compile(r"SUFFICIENT:\s*(yes|no)", re.IGNORECASE)


def synthesize(
    question: str,
    docs: list[dict[str, Any]],
    references: list[dict[str, Any]],
    *,
    budget: Budget,
    max_tokens: int = 1400,
) -> tuple[str, bool]:
    """Final cited synthesis. Returns (markdown_answer, sufficient)."""
    ref_by_doc = {r["doc_id"]: r for r in references}
    blocks: list[str] = []
    for doc in docs:
        canonical = doc.get("canonical_doc_id") or doc.get("doc_id") or ""
        ref = ref_by_doc.get(canonical)
        label = ref["bib_key"] if ref else canonical
        summary = doc.get("summary") or _doc_context(doc)[:500]
        blocks.append(f"[{label}] {doc.get('title') or canonical}\n{summary}")
    evidence_block = "\n\n".join(blocks)
    prompt = (
        "You are writing a concise, well-cited research synthesis answering the QUESTION "
        "from the EVIDENCE below. Cite claims inline using the bracketed keys exactly as "
        "given, e.g. [smith2023]. Only use the provided evidence; do not invent citations. "
        "After the answer, add a line 'SUFFICIENT: yes' if the evidence adequately answers "
        "the question, or 'SUFFICIENT: no' if key aspects remain unsupported.\n\n"
        f"QUESTION: {question}\n\nEVIDENCE:\n{evidence_block}\n\n"
        "Write the answer now (markdown, 2-5 paragraphs), then the SUFFICIENT line."
    )
    try:
        text = generate(prompt, step="synthesis", max_tokens=max_tokens, budget=budget).strip()
    except Exception as exc:  # noqa: BLE001
        log.warning("synthesis failed (%s)", exc)
        return "", True
    match = _SUFFICIENT_RE.search(text)
    sufficient = True
    if match:
        sufficient = match.group(1).lower() == "yes"
        text = _SUFFICIENT_RE.sub("", text).strip()
    return text, sufficient


# --------------------------------------------------------------------------- #
# Optional graph-arm strategy (soft import — no hard dependency)
# --------------------------------------------------------------------------- #
def _resolve_graph_arm(enabled: bool) -> Any | None:
    """Return the graph_arm module if requested AND importable, else None.

    Contract expected of the arm (all optional): a ``gather(query, *, top,
    embed_profile, filters)`` returning ``search``-shaped result dicts. Any import
    or attribute failure degrades silently to plain hybrid retrieval.
    """
    if not enabled:
        return None
    try:
        from graph_layout_rag.query import graph_arm  # type: ignore
    except Exception:  # noqa: BLE001 — arm not present yet; that's fine
        return None
    return graph_arm if hasattr(graph_arm, "gather") else None


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def research(
    question: str,
    *,
    top: int = DEFAULT_TOP,
    max_docs: int = EVIDENCE_DOC_CAP,
    depth: int = 2,
    dry_run: bool = False,
    embed_profile: str | None = None,
    rerank: bool | None = None,
    hybrid: bool = True,
    graph_arm: bool = True,
    tag: str | None = None,
    category: str | None = None,
    source: str | None = None,
    year_min: int | None = None,
    pdf_only: bool = False,
) -> dict[str, Any]:
    """Run the full research pipeline and return a structured result dict."""
    max_docs = min(max_docs, EVIDENCE_DOC_CAP)
    budget = Budget()
    filters = {
        "tag": tag,
        "category": category,
        "source": source,
        "year_min": year_min,
        "pdf_only": pdf_only,
    }
    arm = None if dry_run else _resolve_graph_arm(graph_arm)
    strategy = "graph-arm" if arm is not None else "hybrid"

    # (a) decompose — skipped entirely on dry-run (would cost an LLM call).
    subquestions = [question]
    if not dry_run:
        subs = decompose(question, budget=budget)
        subquestions = [question, *[s for s in subs if s.lower() != question.lower()]][
            :MAX_SUBQUESTIONS
        ]

    # (b) gather
    pool = _gather(
        subquestions,
        top=top,
        embed_profile=embed_profile,
        rerank=rerank,
        hybrid=hybrid,
        filters=filters,
        graph_arm=arm,
    )

    # (c) citation-following expansion (no LLM)
    added = expand_citations(
        pool,
        subquestions,
        top_docs=2 * len(subquestions),
        embed_profile=embed_profile,
        cap=max(0, max_docs - min(len(pool), max_docs)),
    )
    pool = pool + added
    docs = pool[:max_docs]

    result: dict[str, Any] = {
        "question": question,
        "strategy": strategy,
        "subquestions": subquestions,
        "dry_run": dry_run,
    }

    if dry_run:
        result["evidence"] = [_evidence_view(d) for d in docs]
        result["synthesis"] = None
        result["sufficient"] = None
        result["iterations"] = 0
        result["references"] = []
        result["cost"] = budget.to_dict()
        return result

    manifest = _manifest_index()

    # (d) evidence summaries — one short LLM call per unique doc (capped)
    summarized: set[str] = set()
    _summarize_docs(docs, question, budget=budget, summarized=summarized)

    # (e) synthesis + at most one iterate round
    references = _references_for(docs, manifest)
    answer, sufficient = synthesize(question, docs, references, budget=budget)
    iterations = 1

    if depth >= 2 and not sufficient and len(docs) < max_docs:
        titles = [d.get("title") or d.get("doc_id") or "" for d in docs]
        follow = _followup(question, titles, budget=budget)
        if follow:
            extra_pool = _gather(
                [follow],
                top=top,
                embed_profile=embed_profile,
                rerank=rerank,
                hybrid=hybrid,
                filters=filters,
                graph_arm=arm,
            )
            present = {d.get("canonical_doc_id") or d.get("doc_id") for d in docs}
            for r in extra_pool:
                key = r.get("canonical_doc_id") or r.get("doc_id")
                if key and key not in present:
                    docs.append(r)
                    present.add(key)
                    if len(docs) >= max_docs:
                        break
            _summarize_docs(docs, question, budget=budget, summarized=summarized)
            references = _references_for(docs, manifest)
            answer, sufficient = synthesize(question, docs, references, budget=budget)
            iterations = 2

    result["evidence"] = [_evidence_view(d) for d in docs]
    result["references"] = references
    result["synthesis"] = answer
    result["sufficient"] = sufficient
    result["iterations"] = iterations
    result["cost"] = budget.to_dict()
    return result


def _summarize_docs(
    docs: list[dict[str, Any]], question: str, *, budget: Budget, summarized: set[str]
) -> None:
    for doc in docs:
        key = doc.get("canonical_doc_id") or doc.get("doc_id") or ""
        if key in summarized:
            continue
        doc["summary"] = summarize_evidence(doc, question, budget=budget)
        summarized.add(key)


def _references_for(docs: list[dict[str, Any]], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for doc in docs:
        canonical = doc.get("canonical_doc_id") or doc.get("doc_id") or ""
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        refs.append(
            _reference_for(doc.get("doc_id") or canonical, canonical, doc.get("title"), manifest)
        )
    return refs


def _evidence_view(doc: dict[str, Any]) -> dict[str, Any]:
    excerpts = [
        {
            "text": ev.get("excerpt", ""),
            "page": ev.get("page"),
            "section_path": ev.get("section_path", ""),
        }
        for ev in (doc.get("evidence") or [])[:2]
    ]
    return {
        "doc_id": doc.get("doc_id"),
        "canonical_doc_id": doc.get("canonical_doc_id"),
        "title": doc.get("title"),
        "score": doc.get("score"),
        "via_citation": bool(doc.get("via_citation")),
        "excerpts": excerpts
        or (
            [{"text": doc.get("excerpt", ""), "page": doc.get("page")}]
            if doc.get("excerpt")
            else []
        ),
        "summary": doc.get("summary", ""),
    }


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_markdown(result: dict[str, Any]) -> str:
    lines: list[str] = [f"# Research: {result['question']}", ""]
    lines.append(
        f"_strategy: {result['strategy']} · sub-questions: {len(result['subquestions'])} · "
        f"iterations: {result.get('iterations', 0)}_"
    )
    lines.append("")
    lines.append("## Sub-questions")
    for sq in result["subquestions"]:
        lines.append(f"- {sq}")
    lines.append("")

    if result.get("synthesis"):
        lines.append("## Synthesis")
        lines.append(result["synthesis"])
        lines.append("")
        suff = result.get("sufficient")
        if suff is not None:
            lines.append(f"_evidence self-report: {'sufficient' if suff else 'INSUFFICIENT'}_")
            lines.append("")

    lines.append("## Evidence")
    for i, ev in enumerate(result.get("evidence", []), 1):
        via = " (via citation)" if ev.get("via_citation") else ""
        score = ev.get("score")
        lines.append(f"{i}. **{ev.get('title') or ev.get('doc_id')}**{via}  `[{score}]`")
        lines.append(f"   id: {ev.get('canonical_doc_id') or ev.get('doc_id')}")
        if ev.get("summary"):
            lines.append(f"   {ev['summary']}")
        lines.append("")

    if result.get("references"):
        lines.append("## References")
        for ref in result["references"]:
            year = f" ({ref['year']})" if ref.get("year") else ""
            lines.append(f"- `[{ref['bib_key']}]` {ref['title']}{year}")
        lines.append("")

    cost = result.get("cost", {})
    lines.append(
        f"_cost: {cost.get('llm_calls', 0)} LLM calls · "
        f"~{cost.get('input_tokens_est', 0)}in/{cost.get('output_tokens_est', 0)}out tokens · "
        f"~${cost.get('usd_est', 0)}_"
    )
    return "\n".join(lines)
