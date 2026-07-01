"""Read-only corpus/infra health audit for graph-layout-rag experiments."""

from __future__ import annotations

import collections
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from graph_layout_rag.paths import CHUNKS_TABLE, RAW_DIR, profile_index_paths

SEVERITY = {"critical": 3, "warning": 2, "info": 1, "ok": 0}


@dataclass
class Finding:
    severity: str
    code: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)
    remedy: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "detail": self.detail,
            "remedy": self.remedy,
        }


def _load_chunks(profile: str) -> list[dict[str, Any]]:
    import lancedb

    paths = profile_index_paths(profile)
    if not paths.lance_dir.exists():
        return []
    db = lancedb.connect(str(paths.lance_dir))
    names = db.list_tables()
    names = names if isinstance(names, list) else list(getattr(names, "tables", names))
    if CHUNKS_TABLE not in names:
        return []
    cols = ["doc_id", "text", "canonical_sha256"]
    tbl = db.open_table(CHUNKS_TABLE)
    return tbl.search().select(cols).limit(tbl.count_rows()).to_arrow().to_pylist()


def _pdfs_on_disk() -> set[str]:
    pdf_dir = RAW_DIR / "pdf"
    if not pdf_dir.is_dir():
        return set()
    return {p.stem for p in pdf_dir.glob("*.pdf")}


def _manifest_by_id() -> dict[str, Any]:
    """Manifest items keyed by id (``{doc_id: ManifestItem}``).

    Isolated behind a module-level helper so tests can monkeypatch it without a
    real manifest on disk, mirroring the ``_load_chunks``/``_pdfs_on_disk`` seams.
    """
    from graph_layout_rag.manifest import load_manifest, manifest_by_id

    return manifest_by_id(load_manifest())


def audit_chunk_distribution(profile: str, *, min_mean: float = 4.0) -> list[Finding]:
    rows = _load_chunks(profile)
    if not rows:
        return [
            Finding(
                "critical",
                "no_index",
                f"No chunks found for profile {profile!r}.",
                remedy="Run: graph-layout-rag ingest --embed-profile " + profile,
            )
        ]

    cpd = collections.Counter(r["doc_id"] for r in rows)
    n_docs = len(cpd)
    mean = len(rows) / n_docs
    pdfs = _pdfs_on_disk()
    one_chunk_pdf = sorted(d for d, c in cpd.items() if c == 1 and d in pdfs)
    pdf_docs = [d for d in cpd if d in pdfs]
    collapse_rate = (len(one_chunk_pdf) / len(pdf_docs)) if pdf_docs else 0.0

    findings: list[Finding] = [
        Finding(
            "info",
            "chunk_summary",
            f"{len(rows)} chunks / {n_docs} docs (mean {mean:.1f}/doc); "
            f"{sum(1 for c in cpd.values() if c >= 5)} docs >=5 chunks.",
            detail={
                "chunks": len(rows),
                "docs": n_docs,
                "mean_per_doc": round(mean, 2),
                "pdfs_on_disk": len(pdfs),
            },
        )
    ]
    if collapse_rate >= 0.5 and len(pdf_docs) >= 20:
        findings.append(
            Finding(
                "critical",
                "abstract_only_collapse",
                f"{len(one_chunk_pdf)}/{len(pdf_docs)} PDF-backed docs produced only 1 chunk "
                f"({collapse_rate:.0%}) — the index is mostly abstracts, not full text.",
                detail={
                    "one_chunk_pdf_docs": len(one_chunk_pdf),
                    "pdf_docs": len(pdf_docs),
                    "sample": one_chunk_pdf[:10],
                },
                remedy="Re-ingest full text: graph-layout-rag ingest --force --rebuild "
                "--pdf-backend docling after verifying PDFs are present.",
            )
        )
    elif mean < min_mean:
        findings.append(
            Finding(
                "warning",
                "low_chunk_density",
                f"Mean {mean:.1f} chunks/doc is below {min_mean}; corpus may be abstract-heavy.",
                detail={"mean_per_doc": round(mean, 2)},
                remedy="Inspect extraction and consider a forced rebuild with Docling.",
            )
        )
    else:
        findings.append(
            Finding(
                "ok",
                "chunk_density_ok",
                f"Chunk density healthy (mean {mean:.1f}/doc, collapse {collapse_rate:.0%}).",
            )
        )
    return findings


def audit_extraction_fallback(*, log_path: Path | None = None) -> list[Finding]:
    from graph_layout_rag.paths import INGEST_LOG_PATH

    log_path = log_path or INGEST_LOG_PATH
    if not log_path or not log_path.is_file():
        return [Finding("info", "no_ingest_log", "No ingest log to scan for fallbacks.")]
    ok = fallback = 0
    for line in log_path.read_text(errors="ignore").splitlines():
        if "reason=empty_pdf_metadata_fallback" in line:
            fallback += 1
        elif "extracted document=" in line and "status=ok" in line:
            ok += 1
    total = ok + fallback
    if total == 0:
        return [
            Finding("info", "no_extraction_records", "Ingest log has no per-doc extraction lines.")
        ]
    rate = fallback / total
    sev = "critical" if rate >= 0.5 else "warning" if rate >= 0.15 else "ok"
    return [
        Finding(
            sev,
            "extraction_fallback_rate",
            f"{fallback}/{total} extractions fell back to abstract ({rate:.0%}).",
            detail={"fallback": fallback, "total": total, "rate": round(rate, 3)},
            remedy=(
                "Many fallbacks usually mean PDFs were absent or unparseable during ingest; "
                "verify PDFs on disk, then rebuild."
                if sev != "ok"
                else ""
            ),
        )
    ]


def audit_credentials() -> list[Finding]:
    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in {"1", "true"}
    has_key = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    has_vertex = bool(os.getenv("GOOGLE_CLOUD_PROJECT")) and (
        bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
        or (Path.home() / ".config/gcloud/application_default_credentials.json").is_file()
    )
    if has_key or (use_vertex and has_vertex):
        return [Finding("ok", "credentials_ok", "Gemini/Vertex credentials present.")]
    return [
        Finding(
            "warning",
            "missing_credentials",
            "No Gemini API key or Vertex ADC found — cloud embedding/judging/transforms will fail here.",
            detail={"use_vertex": use_vertex, "has_api_key": has_key, "has_vertex_adc": has_vertex},
            remedy="Set GEMINI_API_KEY, or use Vertex ADC. Local Ollama transforms do not need Gemini.",
        )
    ]


def audit_fulltext_stratification(profile: str) -> list[Finding]:
    """Split the indexed corpus into full-text vs metadata-only strata.

    Retrieval quality is capped by how much of the corpus can only be matched on
    its abstract: a ``metadata_only`` doc has no body text in the index, so no
    ranker can lift it past that ceiling. This audit quantifies the ceiling from
    the manifest ``status`` field (composition only — it runs no retrieval; the
    nDCG-per-stratum split lives in ``eval recall``, which already loads the
    index). See gold_validation.py:21,33 for the same status/localPath predicate.
    """
    rows = _load_chunks(profile)
    if not rows:
        return []  # audit_chunk_distribution already emits the no_index critical.
    try:
        by_id = _manifest_by_id()
    except Exception as exc:  # noqa: BLE001 — manifest optional; degrade to info, don't crash the audit.
        return [Finding("info", "no_manifest", f"Manifest unavailable for stratification: {exc}")]

    def stratum(doc_id: str) -> str:
        item = by_id.get(doc_id)
        if item is None:
            return "unkeyed"  # chunk doc_id not present in the manifest at all
        if getattr(item, "status", None) == "ok" and getattr(item, "localPath", None):
            return "full_text"
        if getattr(item, "status", None) == "metadata_only":
            return "metadata_only"
        return getattr(item, "status", None) or "unknown"  # "failed" / ok-without-path

    chunk_strata: collections.Counter[str] = collections.Counter(stratum(r["doc_id"]) for r in rows)
    doc_strata: collections.Counter[str] = collections.Counter()
    for doc_id in {r["doc_id"] for r in rows}:
        doc_strata[stratum(doc_id)] += 1

    n_docs = sum(doc_strata.values())
    n_chunks = sum(chunk_strata.values())
    meta_doc_frac = doc_strata["metadata_only"] / n_docs if n_docs else 0.0
    meta_chunk_frac = chunk_strata["metadata_only"] / n_chunks if n_chunks else 0.0

    findings: list[Finding] = [
        Finding(
            "info",
            "fulltext_stratification",
            f"Indexed docs: {doc_strata['full_text']} full-text / "
            f"{doc_strata['metadata_only']} metadata-only / {doc_strata['unkeyed']} unkeyed "
            f"({meta_doc_frac:.0%} metadata-only docs, {meta_chunk_frac:.0%} of chunks).",
            detail={
                "docs_by_stratum": dict(doc_strata),
                "chunks_by_stratum": dict(chunk_strata),
                "metadata_only_doc_fraction": round(meta_doc_frac, 3),
                "metadata_only_chunk_fraction": round(meta_chunk_frac, 3),
            },
        )
    ]
    if meta_doc_frac >= 0.4:
        findings.append(
            Finding(
                "warning",
                "metadata_only_ceiling",
                f"{meta_doc_frac:.0%} of indexed docs are metadata-only (abstract/title text "
                "only) — a hard retrieval ceiling no ranker can breach for those docs.",
                detail={"metadata_only_doc_fraction": round(meta_doc_frac, 3)},
                remedy="Acquire full-text PDFs for metadata_only stubs with DOIs (Unpaywall / "
                "OpenAlex OA-locations / CORE / venue-targeted Crossref), then re-ingest.",
            )
        )
    return findings


def audit_pool_holes(track: str = "catalog") -> list[Finding]:
    from graph_layout_rag.paths import DATA_DIR

    qrels_path = DATA_DIR / "eval" / "qrels" / track / "qrels.json"
    if not qrels_path.is_file():
        return [Finding("info", "no_qrels", f"No judged qrels for track {track!r} yet.")]
    data = json.loads(qrels_path.read_text())
    cases = data.get("cases", data)
    judged_per_case = [len(v) for v in cases.values()] if isinstance(cases, dict) else []
    if not judged_per_case:
        return [Finding("info", "empty_qrels", f"Qrels for {track!r} has no cases.")]
    mean_judged = sum(judged_per_case) / len(judged_per_case)
    return [
        Finding(
            "info",
            "qrels_summary",
            f"{len(judged_per_case)} judged cases, mean {mean_judged:.0f} judged docs/case for {track!r}.",
            detail={"cases": len(judged_per_case), "mean_judged_per_case": round(mean_judged, 1)},
            remedy="Run `eval diagnostics` for exact hole_rate@k / bpref before trusting nDCG deltas.",
        )
    ]


def run_audit(profile: str, *, track: str = "catalog") -> dict[str, Any]:
    findings: list[Finding] = []
    findings += audit_chunk_distribution(profile)
    findings += audit_fulltext_stratification(profile)
    findings += audit_extraction_fallback()
    findings += audit_credentials()
    findings += audit_pool_holes(track)
    findings.sort(key=lambda f: SEVERITY.get(f.severity, 0), reverse=True)
    worst = findings[0].severity if findings else "ok"
    return {
        "profile": profile,
        "track": track,
        "worst_severity": worst,
        "n_critical": sum(1 for f in findings if f.severity == "critical"),
        "n_warning": sum(1 for f in findings if f.severity == "warning"),
        "findings": [f.to_dict() for f in findings],
    }
