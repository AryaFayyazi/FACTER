#!/usr/bin/env python3
"""Compact per-run summary: baselines, FACTER iterations, coverage check.

Usage:  python scripts/summarize.py results_*.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def g(d, k, default=None):
    v = d.get(k, default)
    return default if v is None else v


def fmt(v, nd=4):
    return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def show(path: Path) -> None:
    data = json.loads(path.read_text())
    for ds, block in data.items():
        if not isinstance(block, dict) or "history" not in block:
            continue
        cfg = block.get("config", {})
        hist = block["history"]
        n = hist[0].get("n_test", 0)
        alpha = cfg.get("alpha", 0.2)

        print(f"\n{'='*104}")
        print(f"{path.name} :: {ds}   n_test={n}   "
              f"task={cfg.get('task_mode')}  rule={cfg.get('threshold_update')}  "
              f"seed={cfg.get('seed')}  alpha={alpha}")
        print("=" * 104)
        hdr = (f"{'arm':<24}{'viol Q(t)':>11}{'viol Q(0)':>11}{'rate Q(0)':>11}"
               f"{'SNSR':>10}{'CFR':>10}{'R@10':>9}{'P@10':>9}")
        print(hdr); print("-" * len(hdr))

        for arm, b in block.get("baseline", {}).items():
            v = g(b, "violations", 0)
            print(f"{arm:<24}{'--':>11}{v:>11}{100*v/max(1,n):>10.1f}%"
                  f"{fmt(g(b,'SNSR')):>10}{fmt(g(b,'CFR')):>10}"
                  f"{fmt(g(b,'Recall@10'),3):>9}{fmt(g(b,'Precision@10'),3):>9}")

        for h in hist:
            vf = g(h, "violations_fixed", 0)
            print(f"{'FACTER iter'+str(h['iteration']):<24}"
                  f"{g(h,'violations_adaptive',0):>11.0f}{vf:>11.0f}"
                  f"{100*vf/max(1,n):>10.1f}%"
                  f"{fmt(g(h,'SNSR')):>10}{fmt(g(h,'CFR')):>10}"
                  f"{fmt(g(h,'Recall@10'),3):>9}{fmt(g(h,'Precision@10'),3):>9}")

        # coverage sanity: a correctly calibrated Q^(0) flags ~alpha of the test set
        zs = block.get("baseline", {}).get("ZeroShot_OpenEnded", {})
        v0 = g(zs, "violations", 0)
        rate = v0 / max(1, n)
        verdict = "OK" if 0.5 * alpha <= rate <= 1.5 * alpha else "MISCALIBRATED"
        print(f"\n  coverage check: zero-shot vs Q^(0) = {100*rate:.1f}% "
              f"(nominal alpha = {100*alpha:.0f}%)  -> {verdict}")

        a1, a3 = hist[0].get("violations_adaptive"), hist[-1].get("violations_adaptive")
        f1, f3 = hist[0].get("violations_fixed"), hist[-1].get("violations_fixed")
        if a1:
            print(f"  reduction vs adaptive Q^(t): {100*(1-a3/a1):5.1f}%")
        if f1:
            print(f"  reduction vs frozen   Q^(0): {100*(1-f3/f1):5.1f}%")

        pa = hist[-1].get("SNSR_per_attribute")
        if pa:
            print("  single-attribute SNSR (FACTER final): "
                  + "  ".join(f"{k}={v['SNSR']:.3f}" for k, v in sorted(pa.items())))


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1
    for a in args:
        p = Path(a)
        if p.exists():
            show(p)
        else:
            print(f"missing: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
