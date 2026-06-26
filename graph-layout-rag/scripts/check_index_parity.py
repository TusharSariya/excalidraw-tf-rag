#!/usr/bin/env python
"""HG2/HG3 index integrity CLI for the cross-machine embedding ladder.

Examples:
  # HG2 — chunk-count parity between the canonical source and a reembedded rung:
  uv run python scripts/check_index_parity.py parity --source cuda-qwen0.6b-1024 \
      --target cuda-qwen4b-1024

  # HG3 — emit a portable fingerprint on the Mac, then verify the desktop copy:
  uv run python scripts/check_index_parity.py fingerprint --profile cuda-qwen0.6b-1024 \
      --out /tmp/mac.json
  #   (sync, then on the desktop:)
  uv run python scripts/check_index_parity.py verify --profile cuda-qwen0.6b-1024 \
      --expect /tmp/mac.json

Exit code is non-zero on any mismatch so this composes into shell guards.
"""

from __future__ import annotations

import argparse
import json
import sys

from graph_layout_rag.ingest.guards import (
    assert_chunk_parity,
    compare_fingerprints,
    index_fingerprint,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_par = sub.add_parser("parity", help="HG2: assert source/target chunk-count parity")
    p_par.add_argument("--source", required=True)
    p_par.add_argument("--target", required=True)

    p_fp = sub.add_parser("fingerprint", help="emit an index integrity fingerprint")
    p_fp.add_argument("--profile", required=True)
    p_fp.add_argument("--out", default=None, help="write JSON here (default: stdout)")

    p_ver = sub.add_parser("verify", help="HG3: verify this host's index matches a fingerprint")
    p_ver.add_argument("--profile", required=True)
    p_ver.add_argument("--expect", required=True, help="fingerprint JSON from the source host")

    args = ap.parse_args()

    if args.cmd == "parity":
        try:
            n = assert_chunk_parity(args.source, args.target)
        except ValueError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 1
        print(f"OK: {args.source} and {args.target} both hold {n} chunks")
        return 0

    if args.cmd == "fingerprint":
        fp = index_fingerprint(args.profile)
        text = json.dumps(fp, indent=2, sort_keys=True)
        if args.out:
            with open(args.out, "w") as fh:
                fh.write(text + "\n")
            print(f"wrote fingerprint -> {args.out}")
        else:
            print(text)
        return 0

    if args.cmd == "verify":
        with open(args.expect) as fh:
            remote = json.load(fh)
        local = index_fingerprint(args.profile)
        problems = compare_fingerprints(local, remote)
        if problems:
            print(f"FAIL: {args.profile} differs from {args.expect}:", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            return 1
        print(f"OK: {args.profile} matches {args.expect} (chunks + BM25 tree identical)")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
