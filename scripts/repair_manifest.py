#!/usr/bin/env python3
"""One-off manifest metadata repair for rag-literature-rag / graph-layout-rag.

Repairs the DOI-collision clobber corruption in ``data/manifest.json``: a
handful of items ended up with one paper's DOI/authors/URL but a completely
different paper's title/abstract (the bug that used to let ``upsert_item``
merge two dissimilar-titled records that happened to share an inferred key).
The merge-guard fix (``titles_similar`` / ``TITLE_SIMILARITY_THRESHOLD``) is
already in each tool's ``manifest.py`` going forward; this script repairs the
*existing* corrupted data using the same helpers, reused directly from the
tool package (not reimplemented) by re-execing under that tool's own
uv-managed venv, where the package and its pydantic dependency are already
installed.

Steps per tool (see module docstrings for detail):
  1. backup ``data/manifest.json`` before any ``--apply`` write
  2. arXiv title verification: for every item with an arXiv DOI/externalId,
     batch-query the arXiv API and repair title/abstract/authors/year on
     mismatch (arXiv is authoritative for its own DOI)
  3. author-collision triage (recomputed after step 2): identical author
     lists with dissimilar titles -> reported, not auto-fixed
  4. sha256 duplicate collapse: identical-file groups with similar titles are
     merged into the richest record and the rest deleted; dissimilar-title
     groups are reported only
  5. write a human (.md) and machine (.json) report, including the list of
     doc ids that need re-ingest (embedded text changed) or index purge
     (item deleted)

Usage (from repo root, plain system python3 -- it re-execs itself under the
right tool venv automatically):

  python3 scripts/repair_manifest.py --tool rag-literature-rag            # dry-run (default)
  python3 scripts/repair_manifest.py --tool rag-literature-rag --apply    # write
  python3 scripts/repair_manifest.py --self-test                         # offline unit tests
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parent.parent

TOOL_CONFIG = {
    "rag-literature-rag": {
        "dir": REPO_ROOT / "rag-literature-rag",
        "module": "rag_literature_rag.manifest",
    },
    "graph-layout-rag": {
        "dir": REPO_ROOT / "graph-layout-rag",
        "module": "graph_layout_rag.manifest",
    },
}

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_BATCH_SIZE = 100
ARXIV_SLEEP_SECONDS = 3
ARXIV_DOI_RE = re.compile(r"^10\.48550/arxiv\.(.+)$", re.IGNORECASE)
ARXIV_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)
# Validates the extracted id actually looks like an arXiv id before it goes
# anywhere near the API: a handful of manifest externalIds["ArXiv"] entries
# turned out to be bare category codes ("cs", "math", "cs/") rather than real
# ids, and the arXiv API returns HTTP 400 for the *whole batch* if any single
# id in id_list is malformed -- so these must be filtered before batching.
ARXIV_ID_NEW_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
ARXIV_ID_OLD_RE = re.compile(r"^[a-z-]+(\.[A-Z]{2})?/\d{7}(v\d+)?$")
ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


# --------------------------------------------------------------------------
# venv re-exec: manifest.py (ManifestItem, titles_similar, merge_items) lives
# inside each tool's own uv-managed venv along with pydantic. Rather than
# reimplementing those helpers, hop into that interpreter transparently.
# --------------------------------------------------------------------------
def _reexec_in_tool_venv(tool: str) -> None:
    cfg = TOOL_CONFIG[tool]
    venv_python = cfg["dir"] / ".venv" / "bin" / "python3"
    if not venv_python.exists():
        return  # no venv found -- let the subsequent import fail loudly
    try:
        already = Path(sys.executable).resolve() == venv_python.resolve()
    except OSError:
        already = False
    if already:
        return
    os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])


def _now_ts() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


# --------------------------------------------------------------------------
# arXiv id extraction + API querying (pure stdlib, no tool import needed)
# --------------------------------------------------------------------------
def _looks_like_arxiv_id(candidate: str) -> bool:
    return bool(ARXIV_ID_NEW_RE.match(candidate) or ARXIV_ID_OLD_RE.match(candidate))


def extract_arxiv_id(doi: str | None, external_ids: dict[str, str] | None) -> str | None:
    """arXiv id for an item, from its DOI (10.48550/arxiv.<id>) or externalIds.

    Returns None (rather than a malformed string) for anything that doesn't
    match a real arXiv id shape -- see ARXIV_ID_NEW_RE/ARXIV_ID_OLD_RE.
    """
    if doi:
        m = ARXIV_DOI_RE.match(doi.strip())
        if m and _looks_like_arxiv_id(m.group(1)):
            return m.group(1)
    for key in ("ArXiv", "arXiv", "arxiv"):
        val = (external_ids or {}).get(key)
        if val and _looks_like_arxiv_id(val.strip()):
            return val.strip()
    return None


def _strip_version(arxiv_id: str) -> str:
    return ARXIV_VERSION_RE.sub("", arxiv_id)


def _fetch_arxiv_batch(batch: list[str], *, sleep_seconds: float, label: str) -> dict[str, dict]:
    """Single id_list request, with retry/backoff on transient HTTP errors.

    Returns {base_id: record} for whatever the API actually returned. A
    request-level failure (after retries) or an id simply absent from the
    response both just result in that id being missing from the returned
    dict -- callers decide what "missing" means (retry smaller, or report).
    """
    url = f"{ARXIV_API}?id_list={','.join(batch)}&max_results={len(batch)}"
    raw = None
    max_attempts = 4
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                raw = resp.read()
            break
        except urllib.error.URLError as exc:
            transient = isinstance(exc, urllib.error.HTTPError) and exc.code in (
                429,
                500,
                502,
                503,
                504,
            )
            if attempt < max_attempts and (
                transient or not isinstance(exc, urllib.error.HTTPError)
            ):
                backoff = sleep_seconds * (2**attempt)
                print(
                    f"  [arxiv] {label} attempt {attempt} FAILED ({exc}); retrying in {backoff:.0f}s",
                    file=sys.stderr,
                )
                time.sleep(backoff)
                continue
            print(f"  [arxiv] {label} FAILED permanently: {exc}", file=sys.stderr)
    if raw is None:
        return {}
    return _parse_arxiv_feed(raw)


def _parse_arxiv_feed(raw: bytes) -> dict[str, dict]:
    """Parses an arXiv Atom feed response into {base_id: record}.

    Split out from _fetch_arxiv_batch so the id-extraction logic (in
    particular the old-style-id slash handling) is unit-testable without a
    network call -- see self_test().
    """
    out: dict[str, dict] = {}
    root = ET.fromstring(raw)
    for entry in root.findall("atom:entry", ATOM_NS):
        entry_id_el = entry.find("atom:id", ATOM_NS)
        entry_id = entry_id_el.text.strip() if entry_id_el is not None and entry_id_el.text else ""
        if "/api/errors" in entry_id:
            continue  # malformed/invalid id marker entry
        # http://arxiv.org/abs/2310.11511v1 -> 2310.11511v1 -> 2310.11511
        # Old-style ids ("cs/0406020") contain their own slash, so this must
        # split on the "/abs/" marker, not the last "/" -- rsplit("/", 1)
        # would otherwise strip the category prefix and collide every
        # "<cat>/<number>" id down to just "<number>".
        raw_id = (
            entry_id.split("/abs/", 1)[-1] if "/abs/" in entry_id else entry_id.rsplit("/", 1)[-1]
        )
        base_id = _strip_version(raw_id)

        title_el = entry.find("atom:title", ATOM_NS)
        summary_el = entry.find("atom:summary", ATOM_NS)
        published_el = entry.find("atom:published", ATOM_NS)
        authors = [
            name_el.text.strip()
            for author_el in entry.findall("atom:author", ATOM_NS)
            for name_el in [author_el.find("atom:name", ATOM_NS)]
            if name_el is not None and name_el.text
        ]
        title = re.sub(r"\s+", " ", (title_el.text or "").strip()) if title_el is not None else ""
        summary = (
            re.sub(r"\s+", " ", (summary_el.text or "").strip()) if summary_el is not None else ""
        )
        year = None
        if published_el is not None and published_el.text:
            try:
                year = int(published_el.text[:4])
            except ValueError:
                year = None

        out[base_id] = {
            "title": title,
            "summary": summary,
            "authors": authors,
            "year": year,
            "raw_id": raw_id,
        }
    return out


def fetch_arxiv_records(
    arxiv_ids: list[str], *, sleep_seconds: float = ARXIV_SLEEP_SECONDS
) -> dict[str, dict]:
    """Batch-query the arXiv API. Returns {base_id (no version): record}.

    Ids missing from a full-size (100-wide) batch response are re-queried in
    a smaller batch: the arXiv API has been observed to silently drop a
    handful of old-style slash ids ("cs/0406020") when they're mixed with
    many new-style dotted ids in one large id_list, even though the same id
    resolves fine alone -- so "missing after batch 1" is not yet reportable
    as invalid/withdrawn, only "still missing after the narrow retry" is.
    Ids still missing after that are left absent -- caller reports those.
    """
    unique_ids = sorted({aid for aid in arxiv_ids if aid})
    out: dict[str, dict] = {}
    batches = [
        unique_ids[i : i + ARXIV_BATCH_SIZE] for i in range(0, len(unique_ids), ARXIV_BATCH_SIZE)
    ]
    for batch_num, batch in enumerate(batches):
        label = f"batch {batch_num + 1}/{len(batches)}"
        out.update(_fetch_arxiv_batch(batch, sleep_seconds=sleep_seconds, label=label))
        if batch_num < len(batches) - 1:
            time.sleep(sleep_seconds)

    still_missing = sorted({_strip_version(aid) for aid in unique_ids} - out.keys())
    if still_missing:
        print(
            f"  [arxiv] {len(still_missing)} id(s) missing after full-size batches; retrying narrow",
            file=sys.stderr,
        )
        narrow_batches = [still_missing[i : i + 20] for i in range(0, len(still_missing), 20)]
        for batch_num, batch in enumerate(narrow_batches):
            time.sleep(sleep_seconds)
            label = f"narrow-retry {batch_num + 1}/{len(narrow_batches)}"
            out.update(_fetch_arxiv_batch(batch, sleep_seconds=sleep_seconds, label=label))

    return out


# --------------------------------------------------------------------------
# Pure-logic helpers used by both the real run and --self-test. All take the
# tool's own titles_similar/merge_items as an argument so this file never has
# to reimplement (or risk drifting from) that logic.
# --------------------------------------------------------------------------
def plan_arxiv_repairs(items, titles_similar, arxiv_records: dict[str, dict]):
    """Returns (repairs, invalid) -- lists of plan dicts. Does not mutate items."""
    repairs = []
    invalid = []
    for item in items:
        arxiv_id = extract_arxiv_id(item.doi, item.externalIds)
        if not arxiv_id:
            continue
        base_id = _strip_version(arxiv_id)
        record = arxiv_records.get(base_id)
        if record is None:
            invalid.append(
                {
                    "id": item.id,
                    "arxiv_id": arxiv_id,
                    "reason": "no arXiv entry returned (invalid/withdrawn id)",
                }
            )
            continue
        if not record["title"]:
            invalid.append(
                {"id": item.id, "arxiv_id": arxiv_id, "reason": "arXiv entry had empty title"}
            )
            continue
        if titles_similar(item.title, record["title"]):
            continue
        repairs.append(
            {
                "id": item.id,
                "arxiv_id": arxiv_id,
                "old_title": item.title,
                "new_title": record["title"],
                "old_authors": list(item.authors),
                "new_authors": record["authors"] or list(item.authors),
                "old_year": item.year,
                "new_year": record["year"] if record["year"] is not None else item.year,
                "new_abstract": record["summary"] or item.abstract,
            }
        )
    return repairs, invalid


def apply_arxiv_repairs(items, repairs: list[dict]) -> None:
    by_id = {item.id: item for item in items}
    for r in repairs:
        item = by_id.get(r["id"])
        if item is None:
            continue
        item.title = r["new_title"]
        item.authors = r["new_authors"]
        item.year = r["new_year"]
        item.abstract = r["new_abstract"]
        item.abstractSource = "arxiv-repair"


def find_author_collision_groups(items, titles_similar):
    """Groups of items sharing an identical non-empty author list but with
    at least one pairwise-dissimilar title. Report-only, never auto-fixed."""
    by_authors: dict[tuple[str, ...], list] = defaultdict(list)
    for item in items:
        if item.authors:
            by_authors[tuple(item.authors)].append(item)

    groups = []
    for authors, group_items in by_authors.items():
        if len(group_items) < 2:
            continue
        titles = [gi.title for gi in group_items]
        first = titles[0]
        if any(not titles_similar(first, t) for t in titles[1:]):
            groups.append(
                {
                    "authors": list(authors),
                    "ids": [gi.id for gi in group_items],
                    "titles": titles,
                }
            )
    return groups


def _richness_key(item):
    status_rank = {"failed": 0, "metadata_only": 1, "ok": 2}
    return (
        -status_rank.get(item.status, 0),
        0 if item.doi else 1,
        0 if item.abstract else 1,
        len(item.id),
        item.id,
    )


def plan_sha256_collapse(items, titles_similar):
    """Groups items by non-empty sha256. Similar-titled groups -> collapse
    plan (keep richest, drop rest). Dissimilar -> triage-only report."""
    by_sha: dict[str, list] = defaultdict(list)
    for item in items:
        if item.sha256:
            by_sha[item.sha256].append(item)

    collapses = []
    triage = []
    for sha, group_items in by_sha.items():
        if len(group_items) < 2:
            continue
        titles = [gi.title for gi in group_items]
        first = titles[0]
        all_similar = all(titles_similar(first, t) for t in titles[1:])
        if all_similar:
            ordered = sorted(group_items, key=_richness_key)
            keep = ordered[0]
            remove = ordered[1:]
            collapses.append(
                {
                    "sha256": sha,
                    "keep_id": keep.id,
                    "remove_ids": [r.id for r in remove],
                    "titles": titles,
                }
            )
        else:
            triage.append({"sha256": sha, "ids": [gi.id for gi in group_items], "titles": titles})
    return collapses, triage


def apply_sha256_collapses(manifest, collapses: list[dict], merge_items) -> None:
    by_id = {item.id: item for item in manifest.items}
    for c in collapses:
        keep = by_id.get(c["keep_id"])
        if keep is None:
            continue
        merged = keep
        for rid in c["remove_ids"]:
            other = by_id.get(rid)
            if other is None:
                continue
            merged = merge_items(merged, other, check_title_conflict=False)
        merged.id = c["keep_id"]
        manifest.items = [
            (merged if item.id == c["keep_id"] else item)
            for item in manifest.items
            if item.id not in c["remove_ids"]
        ]


# --------------------------------------------------------------------------
# Report writing
# --------------------------------------------------------------------------
def write_reports(
    *,
    tool: str,
    data_dir: Path,
    ts: str,
    repairs: list[dict],
    invalid: list[dict],
    collision_groups: list[dict],
    collapses: list[dict],
    dup_triage: list[dict],
    applied: bool,
) -> tuple[Path, Path]:
    reingest_ids = sorted({r["id"] for r in repairs} | {c["keep_id"] for c in collapses})
    removed_ids = sorted({rid for c in collapses for rid in c["remove_ids"]})

    payload = {
        "tool": tool,
        "generatedAt": ts,
        "applied": applied,
        "arxiv_title_repairs": repairs,
        "arxiv_invalid_ids": invalid,
        "author_collision_triage": collision_groups,
        "sha256_collapses": collapses,
        "sha256_dissimilar_triage": dup_triage,
        "reingest_doc_ids": reingest_ids,
        "removed_doc_ids": removed_ids,
    }
    json_path = data_dir / f"manifest-repair-report-{ts}.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# Manifest repair report -- {tool}",
        "",
        f"Generated: {ts}  |  Mode: {'APPLIED' if applied else 'DRY RUN'}",
        "",
        "## arXiv title repairs",
        f"{len(repairs)} item(s) repaired from arXiv metadata.",
        "",
    ]
    for r in repairs[:200]:
        lines.append(f"- `{r['id']}` (arXiv {r['arxiv_id']}):")
        lines.append(f"  - old title: {r['old_title']!r}")
        lines.append(f"  - new title: {r['new_title']!r}")
    if len(repairs) > 200:
        lines.append(f"- ... and {len(repairs) - 200} more (see .json)")
    lines += [
        "",
        "## arXiv ids with no valid response (withdrawn / invalid -- not touched)",
        f"{len(invalid)} item(s).",
        "",
    ]
    for i in invalid[:100]:
        lines.append(f"- `{i['id']}` (arXiv {i['arxiv_id']}): {i['reason']}")
    lines += [
        "",
        "## Author-collision triage (needs human review, not auto-fixed)",
        f"{len(collision_groups)} group(s).",
        "",
    ]
    for g in collision_groups[:100]:
        lines.append(f"- ids={g['ids']} authors={g['authors']}")
        for t in g["titles"]:
            lines.append(f"  - {t!r}")
    lines += [
        "",
        "## sha256 duplicate collapses (similar titles -> merged & deleted)",
        f"{len(collapses)} group(s), {len(removed_ids)} item(s) removed.",
        "",
    ]
    for c in collapses[:100]:
        lines.append(f"- keep `{c['keep_id']}`; removed {c['remove_ids']}")
    lines += [
        "",
        "## sha256 duplicate triage (dissimilar titles -- not touched)",
        f"{len(dup_triage)} group(s).",
        "",
    ]
    for t in dup_triage[:100]:
        lines.append(f"- ids={t['ids']}")
        for title in t["titles"]:
            lines.append(f"  - {title!r}")
    lines += [
        "",
        "## Affected doc ids for later re-ingest (embedded text changed)",
        f"{len(reingest_ids)} id(s): {reingest_ids}",
        "",
        "## Removed doc ids (purge orphaned index rows)",
        f"{len(removed_ids)} id(s): {removed_ids}",
        "",
    ]
    md_path = data_dir / f"manifest-repair-report-{ts}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path


# --------------------------------------------------------------------------
# Real run (imports the tool package -- only reached after venv re-exec)
# --------------------------------------------------------------------------
def run_tool(tool: str, *, apply: bool) -> None:
    cfg = TOOL_CONFIG[tool]
    mod = importlib.import_module(cfg["module"])

    manifest = mod.load_manifest()
    manifest_path = mod.MANIFEST_PATH
    print(f"[{tool}] loaded {len(manifest.items)} items from {manifest_path}")

    arxiv_ids = [extract_arxiv_id(item.doi, item.externalIds) for item in manifest.items]
    arxiv_ids = [a for a in arxiv_ids if a]
    print(f"[{tool}] {len(arxiv_ids)} item(s) carry an arXiv id; querying arXiv API...")
    arxiv_records = fetch_arxiv_records(arxiv_ids)
    print(f"[{tool}] arXiv API returned {len(arxiv_records)} record(s)")

    repairs, invalid = plan_arxiv_repairs(manifest.items, mod.titles_similar, arxiv_records)
    print(f"[{tool}] title-mismatch repairs: {len(repairs)}; invalid/withdrawn ids: {len(invalid)}")

    # Apply title repairs to the in-memory manifest unconditionally so every
    # downstream step (author-collision triage, sha256 collapse) is computed
    # against post-repair state, per spec ("recompute AFTER step 2 repairs").
    # This is safe in dry-run too: nothing is persisted unless --apply, and
    # the mutated in-memory manifest is discarded when the process exits.
    apply_arxiv_repairs(manifest.items, repairs)

    collision_groups = find_author_collision_groups(manifest.items, mod.titles_similar)
    print(f"[{tool}] author-collision triage groups: {len(collision_groups)}")

    collapses, dup_triage = plan_sha256_collapse(manifest.items, mod.titles_similar)
    print(
        f"[{tool}] sha256 collapse groups: {len(collapses)}; dissimilar-title triage: {len(dup_triage)}"
    )

    ts = _now_ts()
    data_dir = manifest_path.parent

    if apply:
        backup_path = manifest_path.with_name(f"manifest.json.bak-{ts}")
        shutil.copy2(manifest_path, backup_path)
        print(f"[{tool}] backed up manifest to {backup_path}")

        apply_sha256_collapses(manifest, collapses, mod.merge_items)
        mod.save_manifest(manifest)
        print(f"[{tool}] wrote repaired manifest ({len(manifest.items)} items) to {manifest_path}")

    md_path, json_path = write_reports(
        tool=tool,
        data_dir=data_dir,
        ts=ts,
        repairs=repairs,
        invalid=invalid,
        collision_groups=collision_groups,
        collapses=collapses,
        dup_triage=dup_triage,
        applied=apply,
    )
    print(f"[{tool}] report written: {md_path}")
    print(f"[{tool}] report written: {json_path}")


# --------------------------------------------------------------------------
# Self-test (offline, fixture manifests, no network) -- runs under a tool
# venv (default rag-literature-rag) so it can exercise the real
# ManifestItem/titles_similar/merge_items, not a reimplementation.
# --------------------------------------------------------------------------
def self_test(tool: str) -> None:
    cfg = TOOL_CONFIG[tool]
    mod = importlib.import_module(cfg["module"])
    Item = mod.ManifestItem
    failures = []

    def check(name, cond):
        status = "ok" if cond else "FAIL"
        print(f"  [{status}] {name}")
        if not cond:
            failures.append(name)

    # -- mismatch detection --------------------------------------------------
    careerx = Item(
        id="doi-10-48550-arxiv-2310-11511",
        title="CareerX: A Retrieval-Augmented Generation Framework",
        authors=["Akari Asai"],
        year=2023,
        source="openalex",
        url="https://arxiv.org/pdf/2310.11511",
        status="ok",
        doi="10.48550/arxiv.2310.11511",
    )
    arxiv_records = {
        "2310.11511": {
            "title": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
            "summary": "abstract text",
            "authors": ["Akari Asai", "Zeqiu Wu"],
            "year": 2023,
            "raw_id": "2310.11511v1",
        }
    }
    repairs, invalid = plan_arxiv_repairs([careerx], mod.titles_similar, arxiv_records)
    check("mismatch detected as repair candidate", len(repairs) == 1)
    check("repair targets correct id", repairs and repairs[0]["id"] == careerx.id)
    check(
        "repaired title looks like Self-RAG",
        repairs and "Self-RAG" in repairs[0]["new_title"],
    )

    # -- no false positive on a matching title -------------------------------
    matching = Item(
        id="x",
        title="Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
        authors=[],
        source="openalex",
        url="u",
        status="ok",
        doi="10.48550/arxiv.2310.11511",
    )
    repairs2, _ = plan_arxiv_repairs([matching], mod.titles_similar, arxiv_records)
    check("matching title -> no repair", len(repairs2) == 0)

    # -- invalid/withdrawn id -------------------------------------------------
    missing = Item(
        id="y",
        title="Some Paper",
        authors=[],
        source="s",
        url="u",
        status="ok",
        doi="10.48550/arxiv.9999.99999",
    )
    _, invalid2 = plan_arxiv_repairs([missing], mod.titles_similar, {})
    check("id absent from arXiv response -> reported invalid, not repaired", len(invalid2) == 1)

    # -- sha256 dup collapse: canonical selection ----------------------------
    rich = Item(
        id="b-longer-id",
        title="Paper Alpha",
        authors=["A"],
        source="s",
        url="u",
        status="ok",
        doi="10.1/alpha",
        abstract="has abstract",
        sha256="deadbeef",
    )
    poor = Item(
        id="a",
        title="Paper Alpha",
        authors=["A"],
        source="s",
        url="u",
        status="metadata_only",
        sha256="deadbeef",
    )
    collapses, triage = plan_sha256_collapse([poor, rich], mod.titles_similar)
    check("dup group with similar titles -> one collapse group", len(collapses) == 1)
    check(
        "richest record (ok status, has doi/abstract) kept",
        collapses and collapses[0]["keep_id"] == "b-longer-id",
    )
    check("poorer record scheduled for removal", collapses and collapses[0]["remove_ids"] == ["a"])
    check("no dissimilar-title triage from this group", len(triage) == 0)

    # -- sha256 dup group with dissimilar titles -> triage only -------------
    other1 = Item(
        id="c1",
        title="Totally Different Paper About Graphs",
        authors=[],
        source="s",
        url="u",
        status="ok",
        sha256="cafebabe",
    )
    other2 = Item(
        id="c2",
        title="Unrelated Neural Network Survey",
        authors=[],
        source="s",
        url="u",
        status="ok",
        sha256="cafebabe",
    )
    collapses3, triage3 = plan_sha256_collapse([other1, other2], mod.titles_similar)
    check("dissimilar-title dup group -> no auto-collapse", len(collapses3) == 0)
    check("dissimilar-title dup group -> triage entry", len(triage3) == 1)

    # -- old-style ("cs/0406020") ids must not collide on the numeric suffix
    fake_feed = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/cs/0406020v1</id>
    <title>Algorithms for Drawing Media</title>
    <summary>sum-a</summary>
    <published>2004-06-16T00:49:40Z</published>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/math/0406020v2</id>
    <title>An Unrelated Math Paper</title>
    <summary>sum-b</summary>
    <published>2004-06-01T00:00:00Z</published>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2310.11511v1</id>
    <title>Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection</title>
    <summary>sum-c</summary>
    <published>2023-10-17T18:18:32Z</published>
  </entry>
</feed>"""
    parsed = _parse_arxiv_feed(fake_feed)
    check(
        "old-style id keeps its category prefix (no collision)",
        "cs/0406020" in parsed and "math/0406020" in parsed,
    )
    check(
        "old-style ids remain distinct records",
        parsed.get("cs/0406020", {}).get("title") != parsed.get("math/0406020", {}).get("title"),
    )
    check("new-style id still parses alongside old-style ids", "2310.11511" in parsed)

    # -- malformed externalIds ("cs", "math", "cs/") are filtered, not queried
    check(
        "bare category code is not treated as an arXiv id",
        extract_arxiv_id(None, {"ArXiv": "cs"}) is None,
    )
    check(
        "trailing-slash category code is not treated as an arXiv id",
        extract_arxiv_id(None, {"ArXiv": "cs/"}) is None,
    )
    check(
        "real old-style externalId is accepted",
        extract_arxiv_id(None, {"ArXiv": "cs/0406020"}) == "cs/0406020",
    )
    check(
        "real new-style DOI is accepted",
        extract_arxiv_id("10.48550/arxiv.2310.11511", None) == "2310.11511",
    )

    # -- dry-run never mutates the passed-in items ---------------------------
    original_title = careerx.title
    _ = plan_arxiv_repairs([careerx], mod.titles_similar, arxiv_records)
    check("planning does not mutate the item in place", careerx.title == original_title)

    if failures:
        print(f"\nSELF-TEST FAILED: {len(failures)} check(s): {failures}")
        sys.exit(1)
    print("\nSELF-TEST OK")


# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tool", choices=sorted(TOOL_CONFIG), help="which tool's manifest to repair"
    )
    parser.add_argument("--apply", action="store_true", help="write changes (default is dry-run)")
    parser.add_argument("--self-test", action="store_true", help="run offline unit tests and exit")
    args = parser.parse_args()

    if args.self_test:
        _reexec_in_tool_venv(args.tool or "rag-literature-rag")
        self_test(args.tool or "rag-literature-rag")
        return

    if not args.tool:
        parser.error("--tool is required (unless --self-test)")

    _reexec_in_tool_venv(args.tool)
    run_tool(args.tool, apply=args.apply)


if __name__ == "__main__":
    main()
