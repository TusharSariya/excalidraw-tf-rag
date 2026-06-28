#!/usr/bin/env python3
"""GATE-3 paired bootstrap CIs on nDCG@10 deltas between strategies.

Reads per-case ndcg@10 from a benchmark run-dir's strategies/catalog--<s>.json
files, pairs by case id, and bootstraps (10k resamples) the mean delta for each
requested pair. Prints a table of strategy nDCG@10 + the key deltas with 95% CI.
"""
import argparse
import glob
import json
import os
import random
from statistics import mean


def load_cases(run_dir: str, strategy: str, track: str = "catalog") -> dict[str, float]:
    path = os.path.join(run_dir, "strategies", f"{track}--{strategy}.json")
    d = json.load(open(path))
    return {c["id"]: float(c["ndcg@10"]) for c in d["cases"]}, float(d["ndcg@10"])


def boot_ci(deltas: list[float], n: int = 10000, seed: int = 7) -> tuple[float, float, float]:
    random.seed(seed)
    k = len(deltas)
    means = []
    for _ in range(n):
        s = sum(deltas[random.randrange(k)] for _ in range(k)) / k
        means.append(s)
    means.sort()
    lo = means[int(0.025 * n)]
    hi = means[int(0.975 * n)]
    return mean(deltas), lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="benchmark run dir (has strategies/)")
    ap.add_argument("--track", default="catalog")
    ap.add_argument("--strategies", nargs="+", required=True,
                    help="strategies present in the run dir to report")
    ap.add_argument("--pairs", nargs="+", default=[],
                    help="A:B deltas to bootstrap (nDCG@10 A - B)")
    args = ap.parse_args()

    per = {}
    agg = {}
    for s in args.strategies:
        try:
            per[s], agg[s] = load_cases(args.run_dir, s, args.track)
        except FileNotFoundError:
            print(f"  (missing: {s})")

    print(f"\n== nDCG@10 ({args.track}, {args.run_dir}) ==")
    for s in args.strategies:
        if s in agg:
            print(f"  {s:24s} {agg[s]:.4f}  (n={len(per[s])})")

    print("\n== paired bootstrap deltas (95% CI, 10k resamples) ==")
    for pair in args.pairs:
        a, b = pair.split(":")
        if a not in per or b not in per:
            print(f"  {pair}: missing strategy")
            continue
        ids = sorted(set(per[a]) & set(per[b]))
        deltas = [per[a][i] - per[b][i] for i in ids]
        m, lo, hi = boot_ci(deltas)
        sig = "" if (lo <= 0 <= hi) else "  *SIG*"
        print(f"  {a} - {b:20s} Δ={m:+.4f}  CI[{lo:+.4f}, {hi:+.4f}]  n={len(ids)}{sig}")


if __name__ == "__main__":
    main()
