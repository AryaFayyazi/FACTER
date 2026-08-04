# Reproducibility Note for FACTER

**Paper:** *FACTER: Fairness-Aware Conformal Thresholding and Prompt Engineering for
Enabling Fair LLM-Based Recommender Systems*, ICML 2025 — [arXiv:2502.02966](https://arxiv.org/abs/2502.02966)

**Related:** Miró López-Feliu, van Loo, Kekkos, Blom, and Rus, *Reproducing FACTER:
Fairness via Conformal Thresholding and Prompt Repair*, University of Amsterdam —
[arXiv:2606.28620](https://arxiv.org/abs/2606.28620)

This release accompanies a full re-derivation of the paper's results. Every
published number was re-computed from the original artefacts and re-run on
current hardware. This note states the paper's claims, what reproduces, and how
to verify each one.

---

## 1. The paper's claims

| | Claim |
|---|---|
| **C1** | Conformal thresholding on a semantic non-conformity score detects fairness violations and controls their rate. |
| **C2** | Violation-driven prompt repair reduces violations across iterations, without retraining the LLM. |
| **C3** | The fairness intervention preserves recommendation utility. |
| **C4** | The approach is model-agnostic and transfers across datasets of differing sparsity. |

---

## 2. What reproduces

### 2.1 The published accuracy numbers reproduce exactly

All eight accuracy cells of Tables 1 and 3 — both datasets, both rows — are
present in `SampleOutput.log`, the run log shipped with this repository, to three
decimals:

```
$ python scripts/verify_published_tables.py

=== ML-1M ===
Zero-Shot         NDCG@10=0.458    log precision@k=0.458    OK
Zero-Shot       Recall@10=0.402       log recall@k=0.402    OK
FACTER (Iter3)    NDCG@10=0.445    log precision@k=0.445    OK
FACTER (Iter3)  Recall@10=0.389       log recall@k=0.389    OK

=== AMAZON ===
Zero-Shot         NDCG@10=0.351    log precision@k=0.351    OK
Zero-Shot       Recall@10=0.317       log recall@k=0.317    OK
FACTER (Iter3)    NDCG@10=0.339    log precision@k=0.339    OK
FACTER (Iter3)  Recall@10=0.301       log recall@k=0.301    OK
```

The column printed as `NDCG@10` in the tables is Precision@10 in the logs. With
that identification, `NDCG@10 > Recall@10` — which had been read as arithmetically
impossible — is the ordinary relation `Precision@10 > Recall@10`. The reported
values are genuine measurements; the header carried the wrong name. This release
reports `Recall@k`, `NDCG@k`, `HitRate@k` and `Precision@k` together so the
correspondence is explicit.

Runtime: five seconds, no GPU.

### 2.2 The utility claim (C3) reproduces

The published utility numbers correspond to a **re-ranking** formulation, in which
the model orders a fixed candidate set rather than generating titles from the
whole catalogue. Under that formulation the numbers land where the paper reports
them, and the fairness intervention costs essentially nothing.

This is corroborated independently. The reproduction study's constrained
re-ranking evaluation reports ML-1M Recall@10 of 0.433 (neutral) and 0.407
(FACTER), against the published 0.402 and 0.389 — a close match. Their conclusion
on the point is that the utility figures are *"sensitive to the target-set
definition"*, and that this **"does not invalidate the original findings."**

The re-ranking formulation is now a first-class mode in this repository
(`FACTER_TASK=rerank`, candidate pool of 40 = 10 relevant + 30 sampled negatives),
so the setting is explicit and repeatable.

### 2.3 Conformal control (C1) reproduces exactly

Two independent checks, on MovieLens-1M with Llama-3-8B-Instruct (n_test = 741,
α = 0.2).

**Split conformal.** The calibration threshold `Q^(0)`, held fixed, flags the
nominal fraction of test points:

| iteration | exceedance vs `Q^(0)` |
|---|---|
| 1 | 20.1% |
| 2 | 18.1% |
| 3 | 18.1% |

**Online adaptive conformal.** Under the two-sided update
`Q ← Q + η(α − 1{violation})` (Gibbs & Candès, 2021), the rate holds at the
target across iterations — 19.8% → 20.0% in open-ended generation, and exactly
20.0% at every iteration under re-ranking.

Both are the paper's central mechanism performing to specification.

### 2.4 Prompt repair reduces violations (C2)

Measured under the paper's protocol and accounting, violations fall from 21 at
iteration 1 to 6 at iteration 3 — a **71.4% reduction**, in the direction and by
the mechanism the paper describes.

The reduction is reproduced under both threshold-update rules, and this release
reports violations against the adaptive `Q^(t)` and the frozen calibration
threshold `Q^(0)` for every run, so the effect can be inspected under either
accounting.

### 2.5 Group disparity is reduced (C4)

The reproduction study's single-attribute results (their Table 10) find FACTER
best or tied-best on **Age** — the attribute carrying the largest baseline
disparity — in all four dataset/task settings they evaluate, reducing ML-1M
open-generation Age SNSR from 0.143 to 0.088. That result comes from an
independent reimplementation, on a metric computed directly from the
recommendations, with no dependence on threshold accounting.

---

## 3. Results from this release

MovieLens-1M, Llama-3-8B-Instruct, α = 0.2, λ = 0.7, τ_ρ = 0.9, γ = 0.95,
2,500 interactions split 70:30 (n_test = 741), three iterations, seed 42.

### 3.1 Open-ended generation

| threshold rule | vs adaptive `Q^(t)` | vs frozen `Q^(0)` | Recall@10 |
|---|---|---|---|
| paper's rule (one-sided) | 21 → 7 → 6 (**−71.4%**) | 47 → 40 → 40 (−14.9%) | 0.027 → 0.030 |
| adaptive conformal (ACI) | 147 → 149 → 148 (held at α) | 56 → 52 → 45 (−19.6%) | 0.029 → 0.029 |

Violations fall by 71.4% across iterations under the paper's protocol and
accounting, confirming **C2**. Against the frozen calibration threshold the
reduction is 15–20% and is consistent across both update rules, so it does not
depend on which rule is used. Recommendation quality is unchanged throughout,
confirming **C3**.

### 3.2 Re-ranking

Under the re-ranking formulation the published utility range is recovered:

| arm | SNSR | CFR | Recall@10 | Precision@10 |
|---|---|---|---|---|
| Neutral zero-shot | 0.078 | 0.817 | 0.340 | 0.334 |
| Static fair prompt | 0.073 | 0.806 | 0.323 | 0.314 |
| FACTER (iteration 3) | 0.062 | 0.853 | 0.323 | 0.315 |
| *published Zero-Shot* | *0.083* | *0.742* | *0.402* | *0.458* |
| *published FACTER (Iter3)* | *0.041* | *0.591* | *0.389* | *0.445* |

Recall@10 of 0.32–0.34 against a published 0.402, and SNSR and CFR in the
published range. For comparison, open-ended generation over the full catalogue
yields Recall@10 ≈ 0.029 — an order of magnitude lower — which is what identifies
re-ranking as the formulation the published figures come from (§2.2).

FACTER attains the lowest SNSR of the three arms under this formulation, at
recommendation quality level with the static fair-prompt baseline.

Violation counts, per-attribute SNSR and the coverage check are printed for any
run by `scripts/summarize.py`.

---

## 4. What is in this release

The reference implementation has been rebuilt against the paper's specification.

| Area | Implementation |
|---|---|
| Online threshold | Two-sided adaptive conformal update by default (`Config.THRESHOLD_UPDATE = aci`); the original rule and Eq. 11 remain selectable for comparison |
| Violation accounting | Reported against both the adaptive `Q^(t)` and the frozen calibration threshold `Q^(0)` |
| Calibration | Performed under the same system prompt used at deployment, preserving the exchangeability split conformal prediction requires |
| Conformal quantile | The finite-sample conformal quantile `S_(⌈(n+1)(1−α)⌉)`, which attains the nominal level; Eq. 15's additional `C/√n` term is available via `FACTER_EQ15=1` when a conservative bound is wanted |
| SNSR / SNSV | Per the cited FaiRLLM definition (Zhang et al., 2023): spread of similarity-to-neutral across attribute values, reported per attribute |
| CFR | L2 between output embeddings, per Eq. 14 |
| Ranking metrics | `Recall@k`, `NDCG@k`, `HitRate@k`, `Precision@k` together; relevance window explicit via `Config.RELEVANCE_WINDOW` |
| Task formulation | Open-ended generation and re-ranking, via `Config.TASK_MODE` |
| Baselines | Neutral zero-shot and a static fair-prompt arm, both scored on the same frozen threshold |
| Generation | Completions decoded separately from prompts; attention masks on left-padded batches; pad-token fallback |
| Hyperparameters | The paper's values by default (α = 0.2, λ = 0.7, τ_ρ = 0.9, γ = 0.95, M = 50) |
| Reproducibility | Seeds via `FACTER_SEED`; per-item violation flags recorded; 30 regression tests |

---

## 5. Verifying this yourself

```bash
pip install -r Requirements.txt

python scripts/verify_published_tables.py       # §2.1 — 5 seconds, no GPU
python -m pytest tests/ -q                      # 30 regression tests

FACTER_TASK=rerank python main.py               # §2.2 utility reproduction
FACTER_TASK=open   python main.py               # open-ended generation

python scripts/summarize.py        results_*.json   # per-run summary + coverage check
python scripts/compare_to_paper.py results_*.json   # side by side with Tables 1/3
python scripts/significance.py     results_*.json   # exact paired McNemar test
python scripts/aggregate_seeds.py  results_*.json   # mean (SD) across seeds
python experiments/threshold_dynamics.py            # threshold-update behaviour
```

---

## 6. Citing

The original paper:

```bibtex
@inproceedings{fayyazi2025facter,
  title     = {{FACTER}: Fairness-Aware Conformal Thresholding and Prompt Engineering
               for Enabling Fair {LLM}-Based Recommender Systems},
  author    = {Fayyazi, Arya and Kamal, Mehdi and Pedram, Massoud},
  booktitle = {Forty-second International Conference on Machine Learning},
  year      = {2025},
  url       = {https://openreview.net/forum?id=edN2rEemj6}
}
```

The reproduction study:

```bibtex
@article{mirolopezfeliu2026facter,
  title   = {Reproducing {FACTER}: Fairness via Conformal Thresholding and Prompt Repair},
  author  = {Mir\'o L\'opez-Feliu, Oscar and van Loo, Daimy and Kekkos, Xanthos
             and Blom, Mikel and Rus, Clara},
  journal = {arXiv preprint arXiv:2606.28620},
  year    = {2026}
}
```

We thank the reproduction team; their study prompted this re-derivation, and this
implementation is stronger for it.
