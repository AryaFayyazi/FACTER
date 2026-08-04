#!/usr/bin/env python3
"""Line the corrected run up against the published tables.

Usage:
    python scripts/compare_to_paper.py results_paper.json results_corrected.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# arXiv:2502.02966, Table 1 (MovieLens-1M).  N_test = 750 (2,500 interactions, 70:30).
PAPER_ML1M = {
    "Zero-Shot":      {"violations": 112, "SNSR": 0.083, "CFR": 0.742, "NDCG@10": 0.458, "Recall@10": 0.402},
    "FACTER (Iter3)": {"violations":   5, "SNSR": 0.041, "CFR": 0.591, "NDCG@10": 0.445, "Recall@10": 0.389},
}
N_TEST_PAPER = 750


def fmt(x, nd=3):
    return "--" if x is None else f"{x:.{nd}f}"


def show(tag: str, path: Path) -> None:
    if not path.exists():
        print(f"\n[{tag}] {path} not found -- run not finished")
        return
    data = json.loads(path.read_text())
    ml = data.get("ml-1m")
    if not ml:
        print(f"\n[{tag}] no ml-1m block")
        return

    zs = ml["baseline"]["ZeroShot_OpenEnded"]
    hist = ml["history"]
    last = hist[-1]
    n = last.get("n_test", zs.get("n_test", 0))

    cfg = ml.get("config", {})
    cfgline = "  ".join(f"{k}={v}" for k, v in cfg.items()) if cfg else ""
    print(f"\n{'='*100}\n[{tag}]  n_test={n}  (paper: {N_TEST_PAPER})")
    if cfgline:
        print(f"  {cfgline}")
    print("=" * 100)
    hdr = f"{'row':<22}{'#Viol Q(t)':>11}{'#Viol Q(0)':>11}{'SNSR':>9}{'CFR':>9}{'NDCG@10':>9}{'Prec@10':>9}{'Recall@10':>10}"
    print(hdr); print("-" * len(hdr))

    print(f"{'Zero-Shot (ours)':<22}{zs.get('violations','--'):>11}{zs.get('violations','--'):>11}"
          f"{fmt(zs.get('SNSR')):>9}{fmt(zs.get('CFR')):>9}{fmt(zs.get('NDCG@10')):>9}"
          f"{fmt(zs.get('Precision@10')):>9}{fmt(zs.get('Recall@10')):>10}")
    p = PAPER_ML1M["Zero-Shot"]
    print(f"{'  paper Zero-Shot':<22}{p['violations']:>11}{'--':>11}{fmt(p['SNSR']):>9}{fmt(p['CFR']):>9}"
          f"{fmt(p['NDCG@10']):>9}{'--':>9}{fmt(p['Recall@10']):>10}")

    for h in hist:
        print(f"{'FACTER iter' + str(h['iteration']):<22}{h.get('violations_adaptive','--'):>11}"
              f"{h.get('violations_fixed','--'):>11}{fmt(h.get('SNSR')):>9}{fmt(h.get('CFR')):>9}"
              f"{fmt(h.get('NDCG@10')):>9}{fmt(h.get('Precision@10')):>9}{fmt(h.get('Recall@10')):>10}")
    p = PAPER_ML1M["FACTER (Iter3)"]
    print(f"{'  paper FACTER Iter3':<22}{p['violations']:>11}{'--':>11}{fmt(p['SNSR']):>9}{fmt(p['CFR']):>9}"
          f"{fmt(p['NDCG@10']):>9}{'--':>9}{fmt(p['Recall@10']):>10}")

    # per-attribute SNSR, comparable with the reproduction study's Table 10
    for label, blk in (("Zero-Shot", zs), (f"FACTER iter{last['iteration']}", last)):
        pa = blk.get("SNSR_per_attribute")
        if pa:
            bits = "  ".join(f"{a}={d['SNSR']:.3f}" for a, d in sorted(pa.items()))
            print(f"  single-attribute SNSR [{label}]: {bits}")

    v1 = hist[0].get("violations_adaptive")
    v3 = last.get("violations_adaptive")
    f1 = hist[0].get("violations_fixed")
    f3 = last.get("violations_fixed")
    if v1:
        print(f"\n  reduction vs adaptive Q^(t): {100*(1-v3/v1):5.1f}%   (paper headline: 95.5%)")
    if f1:
        print(f"  reduction vs frozen   Q^(0): {100*(1-f3/f1):5.1f}%   <- the control")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    for a in sys.argv[1:]:
        show(Path(a).stem.replace("results_", ""), Path(a))
    return 0


if __name__ == "__main__":
    sys.exit(main())
