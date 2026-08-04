#!/usr/bin/env python3
"""Cross-check the published tables against the run log shipped in this repo.

Reproduces, in a few seconds and with no GPU, the two findings in
REPRODUCIBILITY_NOTE.md sections 4 and 5:

  1. Every accuracy cell in Tables 1 and 3 of arXiv:2502.02966 appears in
     SampleOutput.log -- but the column printed as "NDCG@10" is the log's
     `precision@k`, not NDCG.
  2. The "Zero-Shot" row of each table is FACTER's own iteration 1.

Usage:  python scripts/verify_published_tables.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "SampleOutput.log"

# arXiv:2502.02966 -- Table 1 (MovieLens-1M) and Table 3 (Amazon Movies & TV)
PUBLISHED = {
    "ML-1M":  {"Zero-Shot":      {"NDCG@10": 0.458, "Recall@10": 0.402},
               "FACTER (Iter3)": {"NDCG@10": 0.445, "Recall@10": 0.389}},
    "AMAZON": {"Zero-Shot":      {"NDCG@10": 0.351, "Recall@10": 0.317},
               "FACTER (Iter3)": {"NDCG@10": 0.339, "Recall@10": 0.301}},
}

BLOCK = re.compile(
    r'"SNSR":\s*([\d.]+),\s*"SNSV":\s*([\d.]+),\s*"CFR":\s*([\d.]+),'
    r'\s*"ViolationScore":\s*([\d.]+),\s*"precision@k":\s*([\d.]+),'
    r'\s*"recall@k":\s*([\d.]+)'
)


def sections(text: str) -> dict:
    out, names = {}, ("AMAZON", "ML-1M")
    for name in names:
        i = text.index(f"=== Running Experiment on {name} ===")
        j = min([text.find(f"=== Running Experiment on {o} ===") for o in names
                 if o != name and text.find(f"=== Running Experiment on {o} ===") > i]
                or [len(text)])
        out[name] = text[i:j]
    return out


def main() -> int:
    if not LOG.exists():
        print(f"missing {LOG}")
        return 1
    text = LOG.read_text(encoding="utf-8", errors="replace")

    ok = True
    for ds, body in sections(text).items():
        iters = [
            {"SNSR": float(m[0]), "CFR": float(m[2]),
             "precision@k": float(m[4]), "recall@k": float(m[5])}
            for m in BLOCK.findall(body)
        ]
        if len(iters) < 3:
            print(f"[{ds}] expected 3 iterations, found {len(iters)}")
            ok = False
            continue

        # published "Zero-Shot" <- iteration 1 ; "FACTER (Iter3)" <- iteration 3
        pairs = [("Zero-Shot", iters[0]), ("FACTER (Iter3)", iters[2])]
        print(f"\n=== {ds} ===")
        print(f"{'row':<16}{'published':>22}{'log':>26}{'match':>8}")
        for row, it in pairs:
            pub = PUBLISHED[ds][row]
            for pub_col, log_col in (("NDCG@10", "precision@k"),
                                     ("Recall@10", "recall@k")):
                got, want = it[log_col], pub[pub_col]
                hit = abs(got - want) < 5e-4
                ok &= hit
                print(f"{row:<16}{pub_col + '=' + format(want, '.3f'):>22}"
                      f"{log_col + '=' + format(got, '.3f'):>26}"
                      f"{'OK' if hit else 'MISS':>8}")

        print(f"  fairness columns over the same iterations (published direction: down)")
        print(f"    SNSR iter1 -> iter3: {iters[0]['SNSR']:.4f} -> {iters[2]['SNSR']:.4f}")
        print(f"    CFR  iter1 -> iter3: {iters[0]['CFR']:.4f} -> {iters[2]['CFR']:.4f}")

    print("\n" + "=" * 72)
    if ok:
        print("All accuracy cells reproduce from the shipped log.")
        print("The column published as NDCG@10 is precision@10:")
        print("  precision@10 > recall@10 is ordinary, so the reported")
        print("  'NDCG@10 > Recall@10' reflects a column label, not an impossibility.")
    else:
        print("Mismatch -- see rows marked MISS above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
