"""Remap eval gold/qrels doc ids after a manifest dedup collapse.

The manifest repair (scripts/repair_manifest.py) deletes duplicate manifest ids
in favor of a canonical keep_id. Gold cases and qrels files that referenced a
removed id must be remapped or `eval benchmark` fails validation. Dict keys that
collide after remapping (both ids were judged) keep the max relevance.

Usage: python scripts/remap_eval_ids.py --tool <tool-dir> --report <repair-report.json> [--apply]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_mapping(report_path: Path) -> dict[str, str]:
    report = json.loads(report_path.read_text())
    mapping: dict[str, str] = {}
    for collapse in report.get("sha256_collapses", []):
        for removed in collapse["remove_ids"]:
            mapping[removed] = collapse["keep_id"]
    return mapping


def remap(node, mapping: dict[str, str], stats: dict[str, int]):
    if isinstance(node, str):
        if node in mapping:
            stats["values"] += 1
            return mapping[node]
        return node
    if isinstance(node, list):
        return [remap(v, mapping, stats) for v in node]
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            new_key = mapping.get(key, key)
            if new_key != key:
                stats["keys"] += 1
            new_value = remap(value, mapping, stats)
            if new_key in out:
                stats["collisions"] += 1
                if isinstance(new_value, (int, float)) and isinstance(out[new_key], (int, float)):
                    new_value = max(out[new_key], new_value)
            out[new_key] = new_value
        return out
    return node


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True, help="tool directory (e.g. rag-literature-rag)")
    parser.add_argument("--report", required=True, help="manifest repair report JSON")
    parser.add_argument("--apply", action="store_true", help="write changes (default dry-run)")
    args = parser.parse_args()

    tool_dir = Path(args.tool).resolve()
    mapping = build_mapping(Path(args.report))
    print(f"mapping entries: {len(mapping)}")

    gold_files = list((tool_dir / "src").rglob("gold_cases.py"))
    # Only live eval inputs — data/eval/runs/** are historical artifacts, never rewritten.
    qrels_files = list((tool_dir / "data" / "eval" / "qrels").rglob("*.json"))

    for gold in gold_files:
        text = gold.read_text()
        hits = sum(text.count(old) for old in mapping)
        if hits:
            for old, new in mapping.items():
                text = text.replace(f'"{old}"', f'"{new}"').replace(f"'{old}'", f"'{new}'")
            print(f"gold {gold.relative_to(tool_dir)}: {hits} occurrence(s) remapped")
            if args.apply:
                gold.write_text(text)

    for qf in qrels_files:
        raw = qf.read_text()
        if not any(old in raw for old in mapping):
            continue
        stats = {"keys": 0, "values": 0, "collisions": 0}
        try:
            data = remap(json.loads(raw), mapping, stats)
            out_text = json.dumps(data, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:  # JSONL: remap line by line
            lines = [
                json.dumps(remap(json.loads(line), mapping, stats), ensure_ascii=False)
                for line in raw.splitlines()
                if line.strip()
            ]
            out_text = "\n".join(lines) + "\n"
        print(f"qrels {qf.relative_to(tool_dir)}: {stats}")
        if args.apply:
            qf.write_text(out_text)

    print("APPLIED" if args.apply else "DRY-RUN (pass --apply to write)")


if __name__ == "__main__":
    main()
