#!/usr/bin/env python3
"""Paired significance test on violation counts across iterations.

Violation counts at iteration 1 and iteration 3 are measured on the *same* test
rows, so they are paired.  Comparing the totals with an unpaired test throws
away the pairing and can be wildly wrong: totals of 56 and 45 are consistent
with p ~ .001 (if the 45 are a subset of the 56) and with p ~ .27 (if they
barely overlap).  McNemar's test uses the discordant pairs and resolves it.

Usage:  python scripts/significance.py results_rr_corrected.json
"""
from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def report(path: Path, key: str = "is_violation_fixed") -> None:
    data = json.loads(path.read_text())
    for ds, block in data.items():
        if not isinstance(block, dict) or "history" not in block:
            continue
        hist = block["history"]
        first, last = hist[0], hist[-1]
        a = first.get("per_item", {}).get(key)
        z = last.get("per_item", {}).get(key)
        print(f"\n=== {path.name} :: {ds} :: {key} ===")
        if not a or not z or len(a) != len(z):
            print("  per-item flags absent (run predates scripts/significance.py)")
            n1 = first.get("violations_fixed")
            n3 = last.get("violations_fixed")
            if n1 and n3:
                print(f"  totals only: {n1:.0f} -> {n3:.0f} "
                      f"({100*(n3/n1-1):+.1f}%); paired test not possible")
            continue

        both = sum(1 for x, y in zip(a, z) if x and y)
        only1 = sum(1 for x, y in zip(a, z) if x and not y)   # fixed by iter 3
        only3 = sum(1 for x, y in zip(a, z) if y and not x)   # newly violating
        neither = len(a) - both - only1 - only3
        p = mcnemar_exact(only1, only3)

        print(f"  n = {len(a)}   iter{first['iteration']} = {sum(a)}   "
              f"iter{last['iteration']} = {sum(z)}")
        print(f"  both={both}  resolved={only1}  introduced={only3}  neither={neither}")
        print(f"  McNemar exact two-sided p = {p:.4g}"
              f"   {'(significant at .05)' if p < 0.05 else '(not significant)'}")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    for arg in sys.argv[1:]:
        for key in ("is_violation_fixed", "is_violation_adaptive"):
            report(Path(arg), key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
