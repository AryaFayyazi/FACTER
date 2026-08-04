#!/usr/bin/env python3
"""Aggregate repeated-seed runs into mean (SD), and test FACTER vs the baselines.

A 15-20% effect cannot be claimed from a single run. This aggregates several
seeds of the same configuration and reports mean (SD) per arm, matching the
presentation used by the reproduction study (arXiv:2606.28620).

Usage:
    python scripts/aggregate_seeds.py results_rr_corrected_seed*.json
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

ARMS = ("ZeroShot_OpenEnded", "FairZeroShot_Static")
METRICS = ("violations", "SNSR", "CFR", "Recall@10", "Precision@10", "NDCG@10")


def msd(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    if not xs:
        return None
    if len(xs) == 1:
        return f"{xs[0]:.4f}"
    return f"{st.mean(xs):.4f} ({st.stdev(xs):.4f})"


def main() -> int:
    paths = [Path(a) for a in sys.argv[1:]]
    if not paths:
        print(__doc__)
        return 1

    runs = []
    for p in paths:
        if not p.exists():
            print(f"  skip (missing): {p}")
            continue
        runs.append((p, json.loads(p.read_text())))
    if not runs:
        print("no results files found")
        return 1

    datasets = sorted({ds for _, d in runs for ds in d if isinstance(d[ds], dict)})
    for ds in datasets:
        blocks = [d[ds] for _, d in runs if ds in d]
        seeds = [b.get("config", {}).get("seed") for b in blocks]
        print(f"\n{'='*92}\n{ds}   n_seeds={len(blocks)}   seeds={seeds}\n{'='*92}")

        rows = []
        for arm in ARMS:
            vals = {m: [b["baseline"].get(arm, {}).get(m) for b in blocks] for m in METRICS}
            rows.append((arm, vals))

        # FACTER at the final iteration
        fac = {m: [] for m in METRICS}
        fac["violations"] = [b["history"][-1].get("violations_fixed") for b in blocks]
        for m in METRICS:
            if m == "violations":
                continue
            fac[m] = [b["history"][-1].get(m) for b in blocks]
        rows.append(("FACTER (final iter)", fac))

        hdr = f"{'arm':<24}" + "".join(f"{m:>18}" for m in METRICS)
        print(hdr); print("-" * len(hdr))
        for name, vals in rows:
            line = f"{name:<24}"
            for m in METRICS:
                line += f"{str(msd(vals[m])):>18}"
            print(line)

        # the comparison the reproduction study asks for
        fair_v = [v for v in rows[1][1]["violations"] if isinstance(v, (int, float))]
        fac_v = [v for v in fac["violations"] if isinstance(v, (int, float))]
        if fair_v and fac_v and len(fair_v) == len(fac_v):
            diffs = [f - c for f, c in zip(fair_v, fac_v)]
            mean = st.mean(diffs)
            print(f"\n  FairZeroShot - FACTER violations (fixed Q^(0)): "
                  f"mean {mean:+.1f}" +
                  (f" (SD {st.stdev(diffs):.1f}, n={len(diffs)})" if len(diffs) > 1 else ""))
            if len(diffs) > 1 and st.stdev(diffs) > 0:
                t = mean / (st.stdev(diffs) / len(diffs) ** 0.5)
                print(f"  paired t = {t:+.2f} on {len(diffs)-1} df"
                      f"  -> {'FACTER better' if t > 0 else 'static prompt better'}"
                      f" {'(|t|>2, suggestive)' if abs(t) > 2 else '(within noise)'}")
            else:
                print("  -> more seeds needed to separate these arms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
