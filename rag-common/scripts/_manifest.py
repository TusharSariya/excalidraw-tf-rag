#!/usr/bin/env python3
"""Expand backup-manifest.toml into a flat job list for the bash backup/restore harness.

Emits one tab-separated job per line:

    <tool>\t<category>\t<method>\t<s3_subpath>\t<local_path>\t<local_rel>

  method     = tar  | sync | copy
  category   = corpus | indexes | retrieval-indexes | related | extracted | eval | meta
  s3_subpath is the run-id-free logical key, e.g. "indexes/cuda-qwen0.6b-1024"
             (backup_to_s3.sh inserts the <run-id> segment for tar/meta jobs)
  local_path is absolute on this machine.
  local_rel  is the path relative to the tool's data/ dir (e.g. "indexes/cuda-qwen0.6b-1024"),
             stamped into S3 object metadata so restore can rebuild a dead machine.

Globs are expanded against the real filesystem; `tar` jobs are restricted to
directories; `defaults.exclude` patterns are filtered out. This is the single
place that understands the manifest — the bash harness stays dumb.

Requires Python >= 3.11 (tomllib).
"""
from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - clear message for old interpreters
    sys.exit("error: need Python >= 3.11 (tomllib) to read backup-manifest.toml")


def _excluded(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in patterns)


def _expand(data_dir: Path, entries: list[str], exclude: list[str]) -> list[Path]:
    """Expand a list of relative globs/paths under data_dir into existing paths."""
    out: list[Path] = []
    for entry in entries:
        matches = sorted(data_dir.glob(entry)) if any(c in entry for c in "*?[") else [data_dir / entry]
        for p in matches:
            if not p.exists():
                continue
            if _excluded(p.name, exclude):
                continue
            out.append(p)
    return out


def jobs_for_tool(tool: str, cfg: dict, data_root: Path, exclude: list[str]):
    data_dir = (data_root / cfg["data_dir"]).resolve()
    rows: list[tuple[str, str, str, str, str, str]] = []

    def emit(category: str, method: str, s3_subpath: str, local: Path):
        local_rel = str(local.relative_to(data_dir))
        rows.append((tool, category, method, s3_subpath, str(local), local_rel))

    # corpus: sync the directory tree as-is (one object per file)
    for p in _expand(data_dir, cfg.get("corpus", []), exclude):
        if p.is_dir():
            emit("corpus", "sync", f"corpus/{p.name}", p)

    # tar categories: one archive per directory, sub-keyed by directory basename
    for category, key in (
        ("indexes", "indexes"),
        ("retrieval_indexes", "retrieval-indexes"),
        ("related", "related"),
        ("eval", "eval"),
    ):
        for p in _expand(data_dir, cfg.get(category, []), exclude):
            if p.is_dir():
                emit(key, "tar", f"{key}/{p.name}", p)

    # extracted: explicit {model-id = path} mapping
    for model_id, rel in cfg.get("extracted", {}).items():
        p = data_dir / rel
        if p.exists() and p.is_dir() and not _excluded(p.name, exclude):
            emit("extracted", "tar", f"extracted/{model_id}", p)

    # meta: individual files copied verbatim
    for p in _expand(data_dir, cfg.get("meta", []), exclude):
        if p.is_file():
            emit("meta", "copy", f"meta/{p.name}", p)

    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", required=True, help="dir the manifest's data_dir paths are relative to (the tools/ dir)")
    ap.add_argument("--tool", action="append", help="limit to these tools (repeatable)")
    ap.add_argument("--datadirs", action="store_true",
                    help="instead of jobs, print '<tool>\\t<abs data_dir>' for each tool (used by restore)")
    args = ap.parse_args()

    with open(args.manifest, "rb") as fh:
        manifest = tomllib.load(fh)

    exclude = manifest.get("defaults", {}).get("exclude", [])
    data_root = Path(args.data_root).resolve()
    tools = manifest.get("tool", {})

    if args.datadirs:
        for tool, cfg in tools.items():
            if args.tool and tool not in args.tool:
                continue
            print(f"{tool}\t{(data_root / cfg['data_dir']).resolve()}")
        return 0

    wanted = set(args.tool) if args.tool else None
    unknown = (wanted - set(tools)) if wanted else set()
    if unknown:
        sys.exit(f"error: unknown tool(s): {', '.join(sorted(unknown))}; known: {', '.join(sorted(tools))}")

    for tool, cfg in tools.items():
        if wanted and tool not in wanted:
            continue
        for row in jobs_for_tool(tool, cfg, data_root, exclude):
            print("\t".join(row))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
