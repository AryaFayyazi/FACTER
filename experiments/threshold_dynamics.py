#!/usr/bin/env python3
"""Is the violation reduction a property of the outputs, or of the threshold?

This is a controlled, model-free check of the online threshold rule.  Scores are
drawn i.i.d. from a **fixed** distribution across all iterations, so by
construction the recommendations never improve.  Any drop in the violation count
is therefore attributable to the threshold rule alone.

Run:  python experiments/threshold_dynamics.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facter.config import Config  # noqa: E402


def run(rule: str, scores: np.ndarray, q0: float, alpha: float,
        gamma: float, eta: float, n_iter: int, n_per_iter: int):
    """Replay `n_iter` passes over freshly drawn scores under one update rule."""
    q = float(q0)
    per_iter, q_end = [], []
    idx = 0
    for _ in range(n_iter):
        viol_adaptive = viol_fixed = 0
        for _ in range(n_per_iter):
            s = float(scores[idx]); idx += 1
            if s > q:
                viol_adaptive += 1
            if s > q0:
                viol_fixed += 1

            if rule == "aci":
                q = max(0.0, q + eta * ((1.0 if s > q else 0.0) - alpha))
            elif rule == "legacy":
                if s > q:                      # one-sided: only fires upward
                    q = gamma * q + (1.0 - gamma) * s
            elif rule == "paper_eq11":
                if s > q:
                    q = gamma * q + (1.0 - gamma) * min(q, s)
            else:
                raise ValueError(rule)
        per_iter.append((viol_adaptive, viol_fixed))
        q_end.append(q)
    return per_iter, q_end


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-iter", type=int, default=3)
    ap.add_argument("--n-per-iter", type=int, default=750)  # ML-1M test size
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=None,
                    help="override Config.ALPHA (use 0.2 to replay the released config)")
    ap.add_argument("--gamma", type=float, default=None,
                    help="override Config.QUANTILE_DECAY (use 0.92 for the released config)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    alpha = args.alpha if args.alpha is not None else Config.ALPHA
    gamma = args.gamma if args.gamma is not None else Config.QUANTILE_DECAY
    total = args.n_iter * args.n_per_iter

    # A fixed, non-improving score distribution.
    scores = rng.gamma(shape=2.0, scale=0.45, size=total + 10)
    q0 = float(np.quantile(scores, 1.0 - alpha))

    print(__doc__.strip())
    print(f"\nalpha={alpha}  gamma={gamma}  eta={Config.ACI_STEP}")
    print(f"Q^(0)={q0:.4f}   scores are i.i.d. and identically distributed every iteration")
    print(f"(so the ONLY thing that can change a violation count is the threshold)\n")

    hdr = f"{'rule':<12}{'iter':>5}{'viol vs Q^(t)':>15}{'viol vs Q^(0)':>15}{'Q^(t) end':>12}"
    print(hdr)
    print("-" * len(hdr))
    for rule in ("legacy", "paper_eq11", "aci"):
        per_iter, q_end = run(rule, scores, q0, alpha, gamma,
                              Config.ACI_STEP, args.n_iter, args.n_per_iter)
        for i, ((va, vf), qe) in enumerate(zip(per_iter, q_end), start=1):
            print(f"{rule if i == 1 else '':<12}{i:>5}{va:>15}{vf:>15}{qe:>12.4f}")
        drop = 100.0 * (1 - per_iter[-1][0] / max(1, per_iter[0][0]))
        print(f"{'':<12}{'':>5}{'reduction vs Q^(t): ' + format(drop, '.1f') + '%':>30}\n")

    print("Reading: under `legacy` the adaptive count collapses while the fixed-threshold")
    print("count stays flat -- the outputs are unchanged by construction, so that entire")
    print("reduction is threshold movement. `paper_eq11` is an exact no-op on violations.")
    print("`aci` holds the violation rate near alpha, which is the intended behaviour.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
